# aninamer frontend plan

## 结论

前端目标仍然是一个高信息密度的匹配审查工作台，但实现顺序必须先后端契约、再 UI。去掉 plan JSON artifact 后，review 只能依赖数据库中的审查快照、路径生命周期、版本、状态转换、operation 和路径事件。前端不能根据粗略 job summary 拼凑“看起来可审查”的页面。

实施顺序：

1. Phase 0: DB review snapshot、schema、状态转换表、版本/precondition、operation、write-ahead path event、恢复安全契约。
2. Phase 1: 基于 Phase 0 schema 的 mock UI。
3. Phase 2+: 接真实 API，逐步实现 rematch、correction、validate/apply、withdraw、restore。

当前落地状态：

- 已接真实 API/UI：job list、review snapshot、source files、episode targets、subtitle links、move items、mapping evidence、operations、safety summary、用户触发的 rollback eligibility 检查。
- 已落地 proposed-only rematch preview 和 rematch commit；commit 只替换 DB 中的 move plan / review snapshot，不移动文件。
- 已落地第一版 proposed-only manual corrections：支持修改 episode target 的 season/episode range，以及覆盖/清除 subtitle link variant；提交后重新生成 deterministic planned destinations，写入 Operation、job move rows 和 review snapshot。
- 已落地第一版 all-for-apply validation：用户显式触发，检查 proposed move items 的 source/destination 基础可执行性，写入 `validate` Operation，不移动文件。
- 已落地第一版 apply 执行：只消费当前 valid all-for-apply validation，执行前 final recheck 和 destination reservation，逐项写 PathEvent / OperationItemResult，成功后 move items 变 `moved`、job 变 `succeeded`。
- 已落地第一版 withdraw to review 执行：用户显式 rollback eligibility 检查产生短 token 后，后端重新 final check/reservation，将 moved 输出文件移动到与 fail/archive 平级的 `review/` bucket，并写 PathEvent / OperationItemResult。
- 已落地第一版 reopen from review：仅支持 `all` scope，将 withdrawn move items 的 review 文件作为新 source，move items 回到 `proposed`，写 Operation / OperationItemResult，不移动文件。
- 未落地：高级 restore、完整 ReviewSafetyLease/SourceGateLease、episode-target 粒度 reopen。

## 核心目标

- 查看 job、当前工作路径、下一步动作。
- 审查 source file、TMDB episode、LLM mapping、subtitle link、move item 的完整关系。
- 指定 TMDB ID 并预览重匹配 diff。
- 手动修正 season/episode、字幕关联和字幕 variant。
- apply 前做全量可执行验证，且 validation 绑定当前版本和 readiness acknowledgement。
- apply 后逐项查看路径、状态、path event 和错误。
- 将已输出文件回撤到已验证隔离的 `review/` 人工处理区。
- 对记录完整且安全条件满足的 job 提供高级恢复。
- 查看失败 LLM 回复用于分析。

## Phase 0 数据模型

### PathRef

所有路径字段使用结构化 `PathRef`，禁止在 API schema 中混用裸字符串：

- `root_kind`: source / output / archive / review / fail / temp / unknown
- `root_id`
- `rel_path`
- `display_path`
- `normalized_abs_path`
- `realpath`
- `casefold_key`
- `redacted`: bool

服务端必须对 PathRef 做 root containment、path traversal、symlink escape、casefold collision 检查。

API 暴露规则：

- `normalized_abs_path`、`realpath` 可以是 DB-only 字段；对客户端可按配置脱敏。
- `display_path` 始终可用于 UI 展示。
- `redacted=true` 时，客户端不得依赖绝对路径执行操作，只能使用 id / PathRef token。
- `missing/unknown` 路径允许 `normalized_abs_path` 和 `realpath` 为空，但必须保留 root_kind 和历史 display_path。
- `root_id` 必须带 generation，避免 watcher/root 配置变化后误用旧路径。

### FileIdentity

恢复资格需要强 identity，不只是 size：

- `size_bytes`
- `mtime_ns`
- `inode`
- `device`
- `quick_hash`
- `content_hash`
- `identity_strength`: weak / sampled / strong

规则：

- 字幕和小文件应使用 full hash。
- 视频至少使用 size + sampled hash + mtime；必要时支持 full hash。
- 缺少足够 identity 的 item 不能自动 restore，只能进入 review/manual。
- 需要自动 restore / retry / already-at-target 判定的 item 必须具备 `identity_strength=strong`。如果 apply/withdraw 前无法建立 strong identity，eligibility 必须把该 item 标记为 manual-needed，不能用 sampled identity 自动放行。

### SourceFile

记录扫描时的源文件快照：

- `id`
- `job_id`
- `source_id`
- `kind`: video / subtitle / ignored
- `original_path`: PathRef
- `current_path`: PathRef
- `current_path_kind`
- `file_identity`: FileIdentity
- `ext`
- `detected_subtitle_variant`: chs / cht / chi / unknown
- `final_disposition`: output / archive / review / original / missing / unknown
- `ignored_reason`

`ignored` 表示系统规则忽略，和用户手动 `excluded` 分开。

### SubtitleLink

字幕关联是一等模型，不塞进 episode target 数组：

- `id`
- `job_id`
- `episode_target_id`
- `subtitle_source_id`
- `variant_detected`
- `variant_override`
- `status`: proposed / excluded / applied / withdrawn / failed
- `created_by`: llm / user / rematch / reopen_from_review
- `error_code`
- `error_message`

### TMDB data

持久化审查需要的 TMDB 信息：

- selected TMDB id、中文名、原名、年份、国家。
- search candidates。
- season episode counts。
- Season 0 specials。
- 每个目标 episode 的 title、air date、overview 摘要、special/OVA/OAD 标记。

这些信息只用于审查，不进入最终文件名。

### EpisodeTarget

手动修正发生在 episode target 语义层：

- `id`
- `job_id`
- `video_source_id`
- `season`
- `episode_start`
- `episode_end`
- `status`: proposed / excluded / applied / withdrawn / failed / mixed / manual_needed
- `manual`: bool
- `created_by`: llm / user / rematch / reopen_from_review

target 状态只表达语义层状态；路径生命周期以 move item 为准。若 target 下子项状态不同，target 显示 `mixed` / `partial`。

### MappingAttempt / MappingEvidence

成功 LLM raw reply 不落库，但前端必须能解释“为什么这么匹配”。因此要保存验证后的结构化证据：

MappingAttempt：

- `id`
- `job_id`
- `stage`: tmdb_resolve / episode_mapping / rematch / correction
- `created_by`: llm / user / rematch / reopen_from_review
- `tmdb_id`
- `model`
- `prompt_digest`
- `input_digest`
- `attempt_no`
- `status`: succeeded / failed
- `created_at`

MappingEvidence：

- `id`
- `attempt_id`
- `episode_target_id`
- `source_file_id`
- `evidence_kind`: filename_cue / source_order / episode_title / specials_order / ova_oad_hint / manual_override / rematch_diff
- `summary`
- `confidence`: high / medium / low / unknown
- `uncertainty_reason`
- `candidate_refs`

SubtitleLinkEvidence：

- `id`
- `attempt_id`
- `subtitle_link_id`
- `subtitle_source_id`
- `evidence_kind`: basename_match / directory_proximity / language_suffix / manual_override / rematch_diff
- `summary`
- `confidence`

约束：

- 只保存结构化证据和 digest，不保存成功 raw LLM 回复。
- 失败 LLM raw reply 只进入 LLM failure 表，用于后续分析。
- correction/rematch 必须追加新 attempt/evidence，而不是覆盖历史证据。

### MoveItem

move item 是文件级路径生命周期：

- `id`
- `job_id`
- `source_id`
- `episode_target_id`
- `subtitle_link_id`
- `kind`: video / subtitle
- `original_src_path`: PathRef
- `planned_dst_path`: PathRef
- `current_path`: PathRef
- `current_path_kind`: source / output / archive / review / fail / missing / unknown
- `review_path`: PathRef
- `file_identity`: FileIdentity
- `status`: proposed / applying / moved / failed / withdrawn / restoring / restored / restore_failed / excluded / inconsistent / manual_needed / blocked
- `error_code`
- `error_message`
- `applied_at`
- `current_path_updated_at`

`current_path` 是读模型，不是唯一真相。路径变更恢复必须参考 PathEvent。

### PathEvent

PathEvent 是 append-only write-ahead ledger。

每个文件移动或目录 finalize 操作必须按顺序执行：

1. 在同一 DB transaction 中创建 Operation 和 `pending` PathEvent。
2. PathEvent 写入并提交后，才允许执行 filesystem move。
3. filesystem move 成功后更新 `fs_done_at`、`post_identity`、`result`。
4. current path 读模型更新成功后写 `db_done_at`。

这条 write-ahead 顺序是强制要求。禁止先执行 filesystem move 再补 PathEvent。

PathEvent 字段：

- `id`
- `event_sequence`
- `operation_id`
- `job_id`
- `subject_type`: move_item / source_file / finalize
- `move_item_id`
- `source_file_id`
- `finalize_event_id`
- `from_path`: PathRef
- `to_path`: PathRef
- `staging_path`: PathRef
- `reservation_id`
- `move_stage`: direct_rename / staging / finalize
- `pre_identity`: FileIdentity
- `post_identity`: FileIdentity
- `fs_done_at`
- `db_done_at`
- `result`: pending / moved / already_at_target_same_identity / failed / inconsistent / finalized
- `error_code`
- `error_message`

`restore_original_layout_to_source` 和 `restore_original_layout_to_review` 不能只依赖 move item PathEvent。ignored、unmapped、excluded source files 和 archive/finalize 目录级移动都必须有 source_file/finalize 级事件链。eligibility 必须逐个 source file 证明路径事件链完整，才能声称可恢复原目录文件布局。

启动修复只以 pending/incomplete PathEvent 为入口做 from/to identity reconcile：

- to_path 存在且 identity 匹配：补齐读模型。
- staging_path 存在且 identity 匹配：继续 no-overwrite finalize 或标记待人工处理，不能删除 staging 文件。
- from_path 仍存在且 to_path 不存在：标记待重试。
- 两边都不存在或 identity 不匹配：标记 `inconsistent`，进入人工处理。

### ReviewExclusionProof

`review_root_exclusion_verified` 不是 bool，必须是结构化证明：

- `status`: verified / stale / failed / unknown
- `canonical_review_root`: PathRef
- `realpath`
- `device`
- `inode`
- `watcher_instance_id`
- `watcher_config_generation`
- `consumer_chain_config_digest`
- `covered_consumers`: watcher / manual_scan / cli_scan / background_worker
- `matched_exclusion_rule`
- `checked_at`
- `ttl_seconds`
- `failure_reason`

withdraw、reopen 和任何 restore-to-review 执行时必须重新验证该 proof。stale/expired/failed 时禁用相关操作。proof 必须覆盖所有会消费输入目录的入口，包括 watcher、manual scan、CLI scan 和后台 consumer；只证明单个 watcher 不够。

### ReviewSafetyLease / SourceGateLease

一次性 proof 不能作为批量移动期间的长期授权。所有会把文件移动到 review 或 watched source 的 operation 都必须持有 operation-bound lease。

ReviewSafetyLease：

- `id`
- `operation_id`
- `review_exclusion_proof_id`
- `canonical_review_root`: PathRef
- `review_root_device`
- `review_root_inode`
- `watcher_config_generation`
- `consumer_chain_config_digest`
- `covered_consumers`
- `issued_at`
- `expires_at`
- `min_ttl_required_seconds`
- `status`: active / expired / revoked / stale

SourceGateLease：

- `id`
- `operation_id`
- `gate_kind`: watcher_pause / watch_root_disabled / excluded_target
- `watch_root_id`
- `source_gate_proof_id`
- `canonical_source_root`: PathRef
- `source_root_device`
- `source_root_inode`
- `watcher_config_generation`
- `consumer_chain_config_digest`
- `covered_consumers`
- `issued_at`
- `expires_at`
- `status`: active / expired / revoked / stale

规则：

- lease 必须绑定 operation id，不能跨 operation 复用。
- 长操作必须续租或逐项重新验证。
- watcher/root generation、consumer digest、root inode/device 任一变化，lease 立即 stale。
- lease 失效时 operation 停止，未处理 item 标记 blocked，不继续移动。
- rollback eligibility 可以展示 proof/lease 可获取性，但真正执行必须在 operation start 创建或绑定当前 lease。

### Job finalize metadata

job 上应持久化：

- `finalize_status`: none / deleted / archived / skipped
- UI label: `deleted` 显示为“源空目录已移除”
- `archive_path`: PathRef
- `original_source_path`: PathRef
- `current_work_path`: PathRef
- `review_root`: PathRef
- `review_exclusion_proof`: ReviewExclusionProof
- `mapping_version`
- `move_plan_version`
- `move_state_revision`
- `review_revision`
- `last_validation_id`
- `last_validation_status`

## 版本、precondition 和 validation

版本语义：

- `mapping_version`: TMDB、episode target、subtitle link、variant override 等语义变化。
- `move_plan_version`: planned destination 或 move plan 变化。
- `move_state_revision`: current path、move status、operation result 变化。
- `review_revision`: UI payload 总修订，用于 ETag/delta。

`mapping_version` 或 `move_plan_version` 变化会让 apply validation stale。`move_state_revision` 变化不会让语义映射 stale，但会让 apply 执行资格过期：apply 请求必须携带 validation 中记录的 `checked_move_state_revision`，并且 operation start 时必须重新校验路径状态、file identity、目标 reservation。

### ValidationResult

ValidationResult 是持久化资源：

- `validation_id`
- `scope`: all_for_apply / selected_check / retry_apply
- `validated_mapping_version`
- `validated_move_plan_version`
- `checked_move_state_revision`
- `checked_at`
- `status`: current / stale / failed
- `stale_reason`
- `blocking_count`
- `warning_count`
- `affected_refs`
- `errors`
- `warnings`

只有 `scope=all_for_apply` 且 mapping/plan 版本当前的 validation 可以用于 apply。`selected_check` 只是局部检查。`retry_apply` 只允许用于 `apply/retry-failed`，不能用于全量 apply。

Apply 执行前有额外硬门槛：

- 若 `checked_move_state_revision != current_move_state_revision`，前端禁用 apply 并提示重新做“全量应用前验证”。
- 即使 revision 匹配，apply operation start 也必须重新检查每个 proposed source 的 `current_path` 和 FileIdentity。
- apply operation start 必须对完整 destination set 做 reservation，覆盖 parent-not-directory、prefix conflict、symlink escape、casefold collision、target exists、outside-root、cross-device 等问题。
- final recheck 或 reservation 失败时，apply 在移动任何文件前整体拒绝，返回 blocking validation result；不得进入 item-level partial move。

### DestinationReservation

DestinationReservation 是持久化排他资源，不能只是一次预检查。

Reservation 使用全局 canonical destination namespace。不能只按 `(root_id, path_key)` 排他，因为不同 root_id 可能指向同一真实目录、互为父子目录，或因为软链接/挂载别名发生 overlap。

字段：

- `id`
- `operation_id`
- `job_id`
- `root_id`
- `root_generation`
- `root_alias_group_id`
- `path`: PathRef
- `normalized_path_key`
- `casefold_path_key`
- `canonical_destination_key`
- `canonical_parent_key`
- `ancestor_keys`
- `descendant_prefix_key`
- `state`: reserved / committed / released / expired / conflict
- `conflict_code`
- `conflict_ref`
- `created_at`
- `expires_at`
- `released_at`
- `committed_at`

并发语义：

- 所有 apply / retry / withdraw / restore 在 final recheck 之后、创建可执行 PathEvent 之前，必须一次性获取完整目标集合 reservation。
- reservation acquisition 必须在单个 DB transaction 内原子完成，并由唯一/排他约束或等价 serializable locking 强制；单纯先查冲突再插入不符合契约。
- reservation 覆盖跨 job、跨 root_id、跨 root alias/overlap 的 normalized path、casefold path、ancestor/prefix 冲突。
- `canonical_destination_key` 基于 canonical parent realpath、device/inode 和 normalized/casefold relative components 计算；root 互为父子或指向同一 realpath 时必须落入同一 `root_alias_group_id` 或可比较的 canonical namespace。
- ancestor/descendant prefix conflict 必须跨所有 alias/overlap roots 查询，不能只在同一 `root_id/root_generation` 下查询。
- 文件目标与其任意 ancestor/descendant 目标互斥，避免 parent-not-directory 和目录 finalize 竞态。
- reservation 绑定 operation id；operation 成功后标记 committed，失败/取消后释放，崩溃恢复按 expires_at 或 operation terminal status 清理。
- partial operation 中已完成移动的目标 reservation 必须按对应 item/path event 标记 committed 或等价终态；未移动目标才允许释放/过期。
- 目标在 reservation 后被外部进程创建时，filesystem no-overwrite primitive 必须失败；失败 item 记录 `target_exists_after_reservation`，不能覆盖。
- reservation 失败时 operation 整体拒绝，不能创建 pending PathEvent，更不能移动部分文件。

### ReadinessAcknowledgement

影响输出完整性的 warning 必须显式确认，且确认可审计、可失效：

- `id`
- `job_id`
- `warning_code`
- `affected_refs`
- `mapping_version`
- `move_plan_version`
- `acknowledged_by`
- `acknowledged_at`
- `status`: current / stale
- `stale_reason`

warning code、affected set、mapping_version 或 move_plan_version 任一变化，ack 自动 stale。单纯 move_state_revision 变化不应让 ack stale。

### Endpoint preconditions and effects

每个写接口必须声明 required preconditions、成功后的版本变化和失败响应中的 current versions。

| Endpoint | Required preconditions | Success effects | Stale/conflict response |
| --- | --- | --- | --- |
| corrections | `base_review_revision`, no active path operation | increments `mapping_version`, `move_plan_version`, `review_revision` | `412` with current versions |
| match preview | no path lock | no version change | `409/423` if locked |
| match commit | preview token or request payload, base versions current | increments `mapping_version`, `move_plan_version`, `review_revision` | `412` with current versions |
| reopen_from_review | ReviewSafetyLease active, selected complete episode targets, base review/move-state revisions current, current path/FileIdentity recheck | withdrawn target/items become proposed; increments `move_plan_version`, `move_state_revision`, `review_revision` | `412/422/423` |
| validate selected | base versions current | creates `selected_check` ValidationResult; no version change | `412` |
| validate all for apply | base versions current | creates `all_for_apply` ValidationResult; no version change | `412` |
| apply | validation_id exists, scope all_for_apply, validated mapping/plan versions match current, checked move_state revision current, readiness ack current, final path/identity recheck passes, destination reservation succeeds | path operation; increments `move_state_revision`, `review_revision` | `412/422/423` |
| resolve_apply_partial | job in apply_partial, selected resolution actions valid, base review/move-state revisions current; if actions move files to review then ReviewSafetyLease active, final current path/FileIdentity recheck and DestinationReservation pass | moved subsets withdrawn/kept, failed subsets marked manual-needed/excluded/retryable; may return job to planned | `412/422/423` |
| rollback eligibility | explicit user-triggered request, base revisions current | no version change; creates short-lived check result/token only for requested mode/scope | `412/423` |
| withdraw | check_token current, ReviewSafetyLease active, scope matches token, base revisions current, final identity recheck and reservation pass | path operation; increments `move_state_revision`, `review_revision` | `412/422/423` |
| restore_moved_to_source | check_token current, SourceGateLease active, identity and reservation valid | path operation; increments `move_state_revision`, `review_revision` | `412/422/423` |
| restore_original_layout_to_source | check_token current, SourceGateLease active, metadata complete, source/finalize path event chain complete, identity and reservation valid | path operation; increments `move_state_revision`, `review_revision` | `412/422/423` |
| restore_original_layout_to_review | check_token current, ReviewSafetyLease active, metadata complete, source/finalize path event chain complete, identity and reservation valid | path operation; increments `move_state_revision`, `review_revision` | `412/422/423` |

## 状态转换表

### Job transitions

| Operation | From | To running | Success | Partial | Failure | Guard |
| --- | --- | --- | --- | --- | --- | --- |
| plan | pending | planning | planned | n/a | failed | source available |
| rematch | planned/failed/succeeded | rematching | planned | n/a | failed | no path lock |
| corrections | planned/failed | planned | planned | partial op result | failed op | no path lock; proposed targets only |
| validate | planned/succeeded/failed | validating | previous status | n/a | previous status + validation failed | versions current |
| apply | planned | applying | succeeded | apply_partial | failed | valid all_for_apply + final recheck/reservation |
| retry_failed_apply_items | apply_partial | applying | succeeded | apply_partial | failed | only failed/proposed items with current validation |
| resolve_apply_partial | apply_partial | resolving | planned/failed/succeeded summary | apply_partial | failed | every partial item has explicit resolution action |
| withdraw | succeeded/apply_partial/failed | withdrawing | succeeded or restore_partial | restore_partial | restore_failed | ReviewSafetyLease active and item has successful moved/current output event |
| restore_moved_to_source | succeeded/restore_partial | restoring | restored state summary | restore_partial | restore_failed | auto-consumption gate |
| restore_original_layout_to_source | succeeded/restore_partial | restoring | restored state summary | restore_partial | restore_failed | metadata complete + source auto-consumption gate |
| restore_original_layout_to_review | succeeded/restore_partial | restoring | restored state summary | restore_partial | restore_failed | metadata complete + ReviewSafetyLease active |
| reopen_from_review | succeeded/restore_partial/failed | planned | planned | n/a | failed op | complete target selection |

Path-changing operations are mutually exclusive and return `423 Locked` while another path operation is running.

### EpisodeTarget transitions

| Event | From | To |
| --- | --- | --- |
| LLM/rematch creates mapping | none | proposed |
| user excludes | proposed | excluded |
| apply all succeeds for all child moves | proposed | applied |
| apply partial child failure | proposed | mixed |
| withdraw all child moves | applied/mixed | withdrawn |
| resolve apply partial split | mixed | proposed / withdrawn / excluded / manual_needed |
| reopen complete target from review | withdrawn | proposed |
| child states diverge | any | mixed |

First version only allows `reopen_from_review` for complete episode targets. Partial child reopen is not supported; mixed apply failures must first go through `resolve_apply_partial`, which either withdraws moved subsets into complete review targets or splits failed/proposed subsets into clean proposed/manual-needed targets.

### MoveItem transitions

| Event | From | To |
| --- | --- | --- |
| generated | none | proposed |
| excluded by target/link | proposed | excluded |
| apply starts | proposed | applying |
| apply item succeeds | applying | moved |
| apply item fails | applying | failed |
| retry failed apply item starts | failed | applying |
| withdraw starts | moved | restoring |
| withdraw to review succeeds | restoring | withdrawn |
| restore to source/layout starts | moved/withdrawn/restored | restoring |
| restore succeeds | restoring | restored |
| restore fails | restoring | restore_failed |
| crash repair cannot reconcile | any | inconsistent |

`inconsistent` cannot be auto-retied.

## Operation 和幂等性

所有写操作统一使用 Operation。包括 corrections、validate、rematch、apply、withdraw、restore、reopen。

Create operation:

- `POST ...` returns `202 Accepted + operation_id`
- response includes `operation` and `poll_after_ms`

Poll operation:

`GET /api/v1/jobs/{job_id}/operations/{operation_id}`

Optional item result pagination:

`GET /api/v1/jobs/{job_id}/operations/{operation_id}/items?cursor=&limit=`

Operation resource：

- `id`
- `type`: corrections / match_preview / match_commit / validate / apply / retry_failed_apply_items / resolve_apply_partial / withdraw / restore_moved_to_source / restore_original_layout_to_source / restore_original_layout_to_review / reopen_from_review
- `job_id`
- `status`: queued / running / succeeded / partial / failed / canceled
- `request_payload_hash`
- `base_versions`
- `progress`
- `result_summary`
- `error`
- `created_at`
- `started_at`
- `finished_at`
- `poll_after_ms`

OperationItemResult：

- `id`
- `operation_id`
- `subject_kind`: episode_target / move_item / source_file / subtitle_link / path_event
- `subject_id`
- `status`: queued / skipped / succeeded / failed / blocked / inconsistent
- `path_event_id`
- `reservation_id`
- `from_path`: PathRef
- `to_path`: PathRef
- `identity_check`: passed / failed / not_applicable
- `error_code`
- `error_message`
- `retryable`
- `retry_block_reason`
- `result_revision`
- `created_at`

Operation summary 只能展示聚合计数和 latest error；逐项结果以 OperationItemResult 分页为准。

幂等规则：

- `Idempotency-Key` scope 为 job + operation type + user/client。
- 所有 POST 写接口和会生成 token/operation 的 POST 检查接口都必须要求 `Idempotency-Key`，包括 match preview、match commit、validate、apply、retry、resolve partial、withdraw、restore、reopen、readiness acknowledgement 和 rollback eligibility。
- 相同 key + 相同 payload hash 返回原 operation。
- 相同 key + 不同 payload hash 返回 `409 Conflict`。
- in-flight duplicate 返回原 operation，不创建新 operation。
- partial / failed / canceled 的 duplicate 也返回原 operation 和终态。
- precondition/lock 在创建 operation 前失败时不消耗 idempotency key。
- operation summary、PathEvent、item result、restore/audit 相关记录不允许按普通短 TTL 清理；可清理的仅是非审计型临时 preview。
- rematch preview TTL 可以短；apply/withdraw/restore operation 结果必须长期保留，至少覆盖审计和恢复窗口。

### Filesystem move primitive

文件移动原语必须可审计且符合安全约束：

- 默认只允许同 filesystem 的 no-overwrite atomic move。
- validation / eligibility 必须检查 device / mount boundary。
- 跨 device move 默认阻断，不允许高层 `copy + unlink` 伪装成 move。
- 如果未来支持跨 device，必须单独设计 copy-verify-then-safe-retain 流程；在验证完成和用户确认前不得删除源。
- 所有 path-changing operation 在移动前必须对目标集合做 reservation，reservation 失败则不得移动任何文件。
- 禁止使用会覆盖目标的 `os.replace`、普通 POSIX `rename` 覆盖语义或 `shutil.move` 默认语义。
- safe move primitive 必须同时满足 same-device、no-overwrite、no-follow、root-contained；目标存在时只能失败或判定 `already_at_target_same_identity`。
- 若平台不支持直接 no-replace rename，必须使用 operation-owned staging path，写入 staging PathEvent，校验 identity 后再用 no-overwrite finalize 到最终路径；finalize 失败不得覆盖目标，不得删除 staging 文件。
- symlink component、root escape、casefold collision、parent-not-directory 必须在 reservation 和 primitive 执行时双重检查。

### Operation fencing

恢复到 watched source path 的 operation 必须持有 operation 级 fencing token：

- pause lease / disabled watch root / excluded target proof 必须绑定 operation id。
- token 必须有最小剩余 TTL，低于阈值时不得开始批量 restore。
- 长操作必须支持续租或逐项重新校验。
- gate 丢失、lease 过期、watcher config generation 改变时，operation 立即停止，剩余 item 标记为 blocked，不继续移动。
- 移动到 review 的 withdraw、restore-original-layout-to-review、reopen-from-review 同样必须持有 ReviewSafetyLease；不能只依赖操作开始前的一次 proof。

## API common schemas

### SelectionScope

selected scope 必须类型化：

```json
{ "kind": "episode_targets", "ids": [1, 2] }
```

允许：

- `{ "kind": "all" }`
- `{ "kind": "episode_targets", "ids": [...] }`
- `{ "kind": "move_items", "ids": [...] }`
- `{ "kind": "source_files", "ids": [...] }`

服务端必须回显 `resolved_affected_refs`。不同操作只接受合适的 kind。

### ProblemDetails

错误响应统一：

- `code`
- `message`
- `details`
- `affected_refs`
- `field_errors`
- `current_versions`
- `blocking_operation_id`
- `retry_after_ms`
- `request_id`
- `operation_id`

`affected_refs` 使用 `{kind, id}`；`field_errors` 使用 `{path, code, message}`；`current_versions` 包含 mapping/move_plan/move_state/review revision。

HTTP 语义：

- `400`: malformed request
- `409`: business state conflict or idempotency payload hash conflict
- `410`: cursor expired; client must refetch snapshot
- `412`: version/precondition mismatch
- `422`: semantic validation failure
- `423`: operation lock held

## 页面结构

### Jobs 总览

显示字段：

- job id
- original source path
- current work path
- output root
- status
- next action
- move counts by status
- warning count
- last validation status
- finalize status
- archive path present/missing
- review exclusion proof: verified / stale / failed
- rollback capability hints: has moved items / has archive records / has review records / needs safety check
- last rollback check status: unchecked / checked_no_token / stale / expired / failed
- last error
- created/updated time

主操作：

- refresh
- scan now / create job if API supports it
- open review

如果第一版不实现 UI 创建 job，则 Jobs 页只读，并明确扫描由后台 watch/CLI 触发。

### Review 工作台

核心布局：

- 左侧：source index / facet
- 中间：episode target source-vs-target 主审查表
- 右侧：详情 tabs（Item / Match Evidence / Operations / Settings & Safety）

右侧 tabs：

- `Item`: source、TMDB target、subtitle links、move items、path events。
- `Match Evidence`: MappingAttempt、MappingEvidence、SubtitleLinkEvidence、rematch/correction history。
- `Operations`: 当前 operation、历史 operation、item results、PathEvent 摘要、archive/finalize 记录完整性、失败项下一步。
- `Settings & Safety`: source/output/archive/review/fail roots、watcher/root generation、ReviewExclusionProof、ReviewSafetyLease、SourceGateLease、pause lease 状态和刷新入口。

### Operation Center

Operation Center 可以是右侧 tab 或独立页面，但第一版必须可从 Review 工作台进入。

显示：

- 当前运行中的 operation 和 blocking operation。
- 历史 operation 列表：type、status、started/finished、counts、latest error。
- OperationItemResult 分页。
- PathEvent 摘要和 incomplete/pending 事件。
- archive/finalize/source_file 记录完整性。
- rollback eligibility check 历史，只显示 `last_checked_at`、status、stale reason，不显示可复用 token。
- partial operation 的安全下一步。

### Settings & Safety

Settings & Safety 是恢复安全入口，不只是配置展示。

显示：

- source/output/archive/review/fail/temp roots 和 generation。
- watcher 状态、consumer chain digest、覆盖的 consumer。
- review root 是否与 fail 平级、是否被所有 consumer 排除。
- ReviewExclusionProof 当前状态。
- ReviewSafetyLease / SourceGateLease 当前状态、过期时间、绑定 operation。
- source restore gate：watcher pause lease / disabled root / excluded target proof。

操作：

- `刷新 review 区安全证明`。
- `获取/刷新恢复到原路径的安全检查`。
- `查看自动消费链路`。

restore/withdraw 确认框只能消费 Settings & Safety 或 rollback eligibility 产生的当前 check token / safety check token；页面加载不得自动获取 token。operation-bound lease 只能在后端 operation start 创建或绑定。

### Match Overview

Review 工作台顶部保留固定的 Match Overview：

- 当前 selected TMDB id、中文名、年份、候选来源。
- 手输 TMDB id 和候选列表入口。
- 当前 mapping attempt：created_by、model/prompt/input digest、attempt time。
- 结构化证据聚合：filename cues、source order、episode title、specials order、OVA/OAD hint、manual/rematch 标记。
- 风险聚合：low confidence、unmapped videos、unassociated subtitles、episode range、Season 0 specials。
- rematch preview diff 入口和 preserve manual corrections 开关。

Match Overview 只展示结构化证据；不展示成功 LLM raw reply。

### 主表行粒度

主表主粒度是 `episode target` 分组行：

- 父行：一个视频 source + 一个 TMDB target / episode range。
- 子行：该 target 下的视频 move item 和 subtitle link / subtitle move items。
- unmapped videos：source-only 特殊行。
- unassociated subtitles：source-only 特殊行。
- ignored files：独立队列，不混进 apply target。

父行展示：

- target selection checkbox
- episode target id
- source video basename
- compact parent path
- path root badge
- current path summary
- mapped `SxxExx` / range
- range episode count
- start/end episode chips
- TMDB title/air date summary
- special / OVA / OAD marker
- match evidence summary: created_by、confidence、filename cue、source order、OVA/OAD hint
- subtitle count / association status
- planned output summary
- target aggregate status
- validation state
- warning/error badges

多集范围行在详情面板列出范围内每一集的 title、air date、overview 摘要。

子行展示：

- move item selection checkbox
- move id
- kind
- source id
- subtitle link id
- subtitle variant
- path root badge
- original path
- current path
- planned destination path
- review path
- move item status
- last path event / check result
- error code/message

选择模型：

- target selection 和 move-item selection 分开。
- sticky action bar 显示“已选 N 个 episode targets / M 个 move items”。
- 按操作对象过滤可用按钮。
- 折叠父行被选中时，只表示选中 episode target，不隐式选中全部子 move items。
- 子行选择不自动选中父行。
- 跨分页选择必须显式进入 `select all filtered` 模式，并在 filter bar 显示作用范围和总数。
- drill-down/focus 后保留选择，但如果 selected item 不在当前过滤结果中，显示“有 N 个选择项不在当前视图”。
- delta / supersede 后，若已选项被删除、被 rematch supersede、状态变为不可操作，sticky bar 必须进入 stale selection 状态，阻断操作并提供 clear selection / refetch。

表格可用性约束：

- 固定列：selection、source basename、mapped target、status。
- 默认只展开有 warning/error、mixed/partial、unassociated subtitle 的父行。
- 长字段默认折叠，详情在 Item tab 展开。
- 支持虚拟滚动。
- 同名 basename 必须显示 compact parent path 以区分。
- source-only 行默认按分区置顶展示：unmapped videos、unassociated subtitles、ignored files 不混入 season 排序。

### Filter 和 focus

需要 persistent filter bar：

- active facets
- warning/readiness focus
- text search
- target selection count
- move-item selection count
- result count
- clear all

点击 warning 或 readiness count 进入可撤销 focus mode，filter bar 显示来源，如 `Readiness: unassociated subtitles`。

空结果必须显示 `no rows match filters`，提供 clear filters / show all rows。

默认排序：

- `season`
- `episode_start`
- source video order

可切换 source order / status / warning 排序。

## Review readiness

Readiness 面板是 apply 前总控，所有 count 都可点击 drill-down 到 affected rows：

- mapped videos count
- unmapped videos count
- unassociated subtitles count
- excluded count
- ignored files count
- planned output count
- will stay original count
- will archive count
- will move to review count
- blocking errors
- non-blocking warnings
- acknowledgements required

影响输出完整性的非 blocking 项必须有当前 ReadinessAcknowledgement 后，`应用全量验证通过的全部项目` 才可用。

Readiness 面板必须提供实际操作入口：

- `查看需要确认的风险`: 打开只读清单，按 warning code 分组展示 affected rows。
- `确认这些风险并继续`: 创建 ReadinessAcknowledgement。
- `重新确认`: 当 acknowledgement stale 时显示，说明失效原因。
- `撤销确认`: 可选，但如果实现，撤销后 apply 立即禁用。

确认文案必须明确用户接受的影响，例如“这些视频不会输出”“这些字幕不会关联”“这些普通文件会进入 archive”。确认记录绑定 `warning_code + affected_refs + mapping_version + move_plan_version`。

Apply 确认框展示 source disposition：

- 输出文件
- 留在原目录
- 将归档
- 将进入 review
- excluded

## 关键交互

### Rematch

Rematch 分两步：

1. `预览重新匹配`: 生成 diff，不改变版本。
2. `确认应用重新匹配`: 提交 operation，递增 `mapping_version` 和 `move_plan_version`。

默认 scope 为 proposed_only。moved / withdrawn 项不被静默覆盖。丢弃手动修正必须提示。

入口：

- Match Overview 常驻 TMDB 区：当前 selected TMDB id、候选列表、手输 TMDB id、语言/年份摘要。
- 手输 TMDB id 不只出现在 no-candidates 空状态；任何 planned/failed job 都能发起 preview。
- preview diff 必须展示新增/删除/变更 target、字幕关联变化、manual corrections 是否保留。

### Reopen from review

第一版只允许按完整 episode target reopen：

- 当前已实现切片只开放 `all` scope；后续再开放完整 `episode_targets` scope。
- 不允许只 reopen 一个 target 下的部分字幕/视频。
- 以 target 下所有 withdrawn child move items 的 `review_path/current_path` 作为新源。
- operation start 必须持有 ReviewSafetyLease，并重新验证 current path 和 FileIdentity。
- target 和 child move items 回到 proposed。
- 保留 original source path 为历史。
- 递增 `move_plan_version`、`move_state_revision` 和 `review_revision`。
- validation stale。
- `apply_partial` 的 mixed target 不能直接 reopen，必须先 `resolve_apply_partial` 形成完整 withdrawn target。

### Corrections

Corrections 是 Operation，支持 `atomic=true/false`。

前端控件必须明确落位：

- 父行 quick actions:
  - episode target 编辑按钮。
  - exclude / unexclude toggle。
  - 打开字幕关联管理。
- `Item` tab 中的 episode target editor:
  - season select。
  - episode start/end stepper 或 range picker。
  - Season 0 specials 标记。
  - 目标文件名即时预览。
- `Item` tab 中的 subtitle link editor:
  - 未关联字幕列表。
  - attach subtitle。
  - detach subtitle。
  - move subtitle to another target。
  - variant segmented control: chs / cht / chi / detected。
- correction draft panel:
  - 展示待提交 ops。
  - 支持 discard。
  - `提交修正` 创建 corrections operation。

已 moved/applied 的 target 不能直接通过 correction 改输出文件。用户必须先 `回撤到 review`，再 `从 review 重新打开`，然后修正。UI 在已应用项上只显示“先回撤后修正”的入口，不显示直接编辑控件。

`apply_partial` 例外：未移动的 failed/proposed 子项可以在 `resolve_apply_partial` 后进入 proposed/manual-needed 分组，再做 correction。已经 moved 的子项仍必须先回撤到 review，不能直接编辑输出路径。

每个 op 有独立 schema 和 result：

- `update_episode`: target id、season、episode_start、episode_end。
- `attach_subtitle`: target id、subtitle source id。
- `detach_subtitle`: subtitle link id。
- `move_subtitle`: subtitle link id、target id。
- `override_subtitle_variant`: subtitle link id、variant。
- `exclude`: target/source/link id + reason。
- `unexclude`: target/source/link id。

Response:

- `op_results[]`
- `resolved_affected_refs`
- regenerated / superseded move items
- move deltas
- new versions
- validation stale reason

### Validate and apply

操作区按钮：

- `检查所选项`: 局部检查，不产生可 apply validation id。
- `全量应用前验证`: 全 job 验证，产生可 apply validation id。
- `应用全量验证通过的全部项目`: 全 job 操作，不受 selection 影响。

当有选中行时，sticky bar 必须显示：

```text
当前选择不会影响应用范围。
```

Apply 确认弹窗必须明确：

- apply 是 move，不是 copy。
- 将移动多少视频/字幕。
- 不保留源侧副本。
- 源目录 finalize 可能是“源空目录已移除” / archived / skipped。
- 回撤依赖 current path / archive path 仍存在。
- source disposition 分组和预测 finalize 结果。

### Apply partial resolution

apply partial 后，job 进入 `apply_partial`，前端必须显示 resolution panel。没有完成 resolution 前，rematch/corrections/apply all 默认禁用。

panel 分组：

- 已移动成功。
- 未移动且仍可重试。
- 失败且需要人工处理。
- 路径状态不一致。

允许的安全下一步：

- `重试未完成项`: 只对 failed/proposed 且 source identity 当前、destination reservation 通过的 item 创建 `retry_failed_apply_items` operation。
- `回撤已移动项到 review`: 对已经 moved 的 item 走 withdraw_to_review。
- `进入人工处理`: 将 inconsistent / non-retryable item 标记为需要人工处理。
- `整理 partial 状态`: 创建 `resolve_apply_partial` operation，把 mixed target 拆成可继续处理的干净分组。

重试前必须重新做 `全量应用前验证` 或针对 retry scope 的等价 validation，并执行 final path/identity recheck。`inconsistent` item 永远不可自动 retry。

`resolve_apply_partial` request：

- `base_review_revision`
- `base_move_state_revision`
- `actions[]`

允许的 action：

- `withdraw_moved_to_review`: 对 moved 子项执行带 ReviewSafetyLease 的 withdraw。
- `keep_moved_as_applied`: 明确保留已移动输出，只把未完成子项拆出。
- `mark_failed_manual_needed`: failed / non-retryable 子项变为 manual_needed，保留 current source path。
- `exclude_failed`: 用户明确排除 failed 子项并记录原因。
- `split_unmoved_to_proposed`: failed/proposed 子项形成新的完整 proposed target，允许后续 rematch/corrections。

resolution 终态：

- 所有 moved 子项必须处于 moved/applied、withdrawn 或 restore_failed/manual-needed 的显式状态。
- 所有 failed/proposed 子项必须处于 proposed、excluded、manual_needed 或 retryable failed 的显式状态。
- 不允许留下没有 resolution action 的 mixed child。
- 若所有 partial item 都 resolved，job 回到 `planned` 或按现有输出摘要显示 `succeeded_with_manual_items`；rematch/corrections 只对 proposed/manual_needed 未移动分组开放。

### UI status labels

不要直接展示内部状态：

- `withdrawn`: 已回撤到 review 区
- `restored`: 已恢复
- `mixed`: 部分状态
- `inconsistent`: 路径状态不一致，需人工处理
- `excluded`: 已手动排除
- `ignored`: 系统忽略
- `deleted` finalize: 源空目录已移除

## 回撤和恢复设计

用户可见动作：

1. `回撤到 review`
2. `恢复已移动/已回撤文件到原路径`
3. `恢复原目录文件布局到原路径`
4. `恢复原目录文件布局到 review 区`，当原路径自动再消费门槛不满足时使用

### review 目录硬约束

`review/` 和 `fail/` 平级：

```text
<archive-control-root>/
  fail/
  review/
```

安全要求：

- 后端必须把 `review` 加入监控硬性排除。
- API 返回 ReviewExclusionProof。
- UI 显示 verified/stale/failed。
- 未验证时禁用 withdraw_to_review 和 restore-to-review 的执行，只允许打开检查面板。
- 执行时必须创建或绑定 ReviewSafetyLease，并在整个 operation 期间保持有效。
- lease 失效、consumer generation 改变、review root inode/device 改变时立即停止，剩余 item 标记 blocked。
- 目标目录由后端生成唯一路径，如 `review/{job_id}-{operation_id}/...`。
- 不删除文件。

### Auto-consumption gate

所有恢复到 watched source path 的操作都必须满足硬门槛之一：

- 后端可验证 watcher pause lease。
- watch root disabled。
- 目标在 excluded/quarantine/review root 下。

eligibility 和确认框展示 gate/lease id、过期时间、验证时间。不允许用用户口头确认解锁。Settings & Safety 提供获取/刷新 gate 的入口；页面加载不自动获取 gate。

`restore_moved_to_source` 和 `restore_original_layout_to_source` 使用同一门槛。`restore_original_layout_to_review` 不恢复到 watched source path，但必须持有 ReviewSafetyLease。

### Restore request schema

`restore-moved-to-source` 请求：

- `scope`: SelectionScope
- `base_review_revision`
- `base_move_state_revision`
- `check_token`
- `source_gate_check_token`
- optional `pause_lease_id` as source gate evidence only

`restore-original-layout-to-source` 请求：

- `scope`: all / selected source files
- `base_review_revision`
- `base_move_state_revision`
- `check_token`
- `source_gate_check_token`
- optional `pause_lease_id` as source gate evidence only

`restore-original-layout-to-review` 请求：

- `scope`: all / selected source files
- `base_review_revision`
- `base_move_state_revision`
- `check_token`
- `review_safety_check_token`

`check_token` 必须绑定 mode、scope、ids、base revisions、destination endpoint 和 TTL。执行仍重新验证。

### Eligibility and partial handling

eligibility 是 mode-specific：

- withdraw_to_review
- restore_moved_to_source
- restore_original_layout_to_source
- restore_original_layout_to_review

触发规则：

- Jobs 总览、Review payload、页面刷新不得实时计算 eligibility，不得扫盘，不得生成 `check_token`。
- Jobs/Review 只返回 `rollback_capability_hints` 和 `last_rollback_check_summary`，用于提示“需要检查”或展示历史结果。
- `last_rollback_check_summary` 必须使用 `unchecked / checked_no_token / stale / expired / failed` 枚举，不能包含可执行 token，不能用于启用执行按钮。
- 只有用户在某个 mode 的确认面板点击“检查资格”时，前端才调用 `/rollback-eligibility`。
- `/rollback-eligibility` 返回短 TTL `check_token`、destination preview 和 item results；执行按钮只在同一面板内基于当前 token 启用。

返回：

- mode
- status: eligible / partial / ineligible
- checked_at
- ttl_seconds
- check_token
- missing files
- incomplete records
- target conflicts
- identity mismatches
- retryable items
- non-retryable items
- affected move items
- archive restore summary
- destination preview
- source availability summary: current output path / archive path / review path / missing / record incomplete

路径冲突分类：

- prefix conflict
- parent-not-directory
- symlink component
- symlink escape
- casefold collision
- outside-root
- target exists
- hardlink/mount boundary changed

执行前对完整目标集合做 reservation，再逐项 move。

Partial 操作规则：

- `inconsistent` 永远不可自动 retry。
- identity mismatch 不可自动 retry。
- missing source 不可自动 retry。
- target already exists with strong same identity: `already_at_target_same_identity`，视为成功/跳过。
- 只有强 identity 匹配、目标仍空或 already-at-target-same-identity 的 item 可自动 retry。
- sampled identity 不足以自动判定 restore/retry 成功；只能进入人工确认或 review。

item result 包含：

- `retryable`
- `retry_block_reason`
- `next_from_path`
- `next_to_path`
- `expected_identity`

恢复确认面板必须逐项展示：

- item id / source id / episode target id。
- restore source: current output path / archive path / review path / staging path / missing / record incomplete。
- restore destination。
- identity strength 和 identity check 结果。
- destination reservation preview。
- 不能处理的原因和建议下一步：重新检查、回撤到 review、manual-needed。

第一版 partial restore 默认只读展示并阻断执行；可以单独提供“仅回撤可验证项到 review”，且必须列出不会处理的文件清单。

## API 契约

Phase 0 必须产出 Pydantic/OpenAPI schema。

通用要求：

- 所有 POST 写接口和 token-producing 检查接口都必须要求 `Idempotency-Key`。
- Operation 型 POST 返回 `202 Accepted + Operation resource`。
- 失败响应必须带 `current_versions`、`blocking_operation_id` 或可操作的 `field_errors`。

### Review payload

`GET /api/v1/jobs/{job_id}/review`

返回：

- job summary
- source file records
- selected TMDB info
- TMDB candidates
- seasons / episodes / specials
- episode targets
- subtitle links
- mapping attempts and structured evidence summaries
- move items
- path event summary
- warnings
- validation summary
- readiness summary
- readiness acknowledgements
- finalize status
- archive path summary
- review exclusion proof
- rollback capability hints
- last rollback check summary: status unchecked / checked_no_token / stale / expired / failed; last_checked_at
- llm failure summary
- `review_revision`

支持 ETag / If-None-Match。

分页和一致性：

- 初始 payload 可以返回 summary + 第一页主表数据。
- 互相关联集合使用同一个 `snapshot_token` 和 `snapshot_revision`。
- episode target 页必须包含该页父行需要的 subtitle links 和 move items，禁止返回“父行无子行”的不完整页。
- source-only rows 使用独立分区分页。
- 稳定排序必须由服务端返回 sort key。
- 大集合分页端点必须声明 max limit。
- cursor 过期返回 `410 Gone`，客户端全量 refetch。

### Move deltas

`GET /api/v1/jobs/{job_id}/moves/delta?after_revision=&limit=`

返回：

- ordered events: upsert / delete / supersede
- move item payload
- `next_cursor`
- `has_more`
- `snapshot_revision`

事件按 revision + id 单调排序。cursor 过期返回 `410 Gone`。

### Operations

`GET /api/v1/jobs/{job_id}/operations?cursor=&limit=&status=&type=`

`GET /api/v1/jobs/{job_id}/operations/{operation_id}`

`GET /api/v1/jobs/{job_id}/operations/{operation_id}/items?cursor=&limit=`

Operation 主资源只返回 summary/counts/latest error。完整 item results 只走分页 endpoint，避免双通道不一致。

Operation list 支撑 Operation Center，不依赖 plan JSON artifact 或单独 log 文件。

### Corrections API

`POST /api/v1/jobs/{job_id}/corrections`

Required headers:

- `Idempotency-Key`

Request:

- `base_review_revision`
- `atomic`
- `ops[]`

Response:

- `200 OK`
- 同步返回 committed correction summary；完整操作记录仍写入 Operation，可通过 operations endpoint 查看。

第一版只支持：

- `update_episode`: 修改现有 proposed episode target 的 season / episode_start / episode_end。
- `override_subtitle_variant`: 修改现有 proposed subtitle link 的 `variant_override`，`detected` 表示清除 override 并回到检测值。
- `atomic=false` 暂不支持；第一版要求 `atomic=true`，否则后端返回 `422 unsupported_correction_mode`。
- 若任一当前 move item 不是 `proposed`，后端直接拒绝，不扫盘、不移动文件。

### Rematch API

`POST /api/v1/jobs/{job_id}/match/preview`

Required headers:

- `Idempotency-Key`

Request:

- forced TMDB id
- scope
- preserve manual corrections flag
- base versions

Response:

- `202 Accepted`
- Operation resource
- completed operation result includes preview token, diff summary, affected refs, stale validation notice

Preview token 绑定 tmdb id、scope、base versions、input digest、manual preservation mode 和 TTL。

`POST /api/v1/jobs/{job_id}/match/commit`

Required headers:

- `Idempotency-Key`

Request:

- preview token or full request payload
- base versions

Response:

- `202 Accepted`
- Operation resource

### Validate API

`POST /api/v1/jobs/{job_id}/validate`

Required headers:

- `Idempotency-Key`

Request:

- `scope`: all_for_apply / selected_check / retry_apply
- SelectionScope when selected_check
- base versions

Response:

- `200 OK`
- 同步返回 ValidationResult；完整结果也写入 `validate` Operation 的 `response_payload`。

第一版只支持：

- `all_for_apply`: 检查所有 proposed move items。
- `selected_check`: 检查 `all` 或指定 `move_items`，但不产生可执行 apply。
- `retry_apply`: schema 保留，当前返回 `422 unsupported_validation_scope`。
- 检查项包括 source missing、destination exists、outside output root、parent-not-directory、同批 destination collision。

### Reopen API

`POST /api/v1/jobs/{job_id}/reopen-from-review`

Required headers:

- `Idempotency-Key`

Request:

- `scope`: all withdrawn targets or SelectionScope with complete episode target ids
- `base_review_revision`
- `base_move_state_revision`
- `review_safety_check_token`

Response:

- `202 Accepted`
- Operation resource

### Readiness acknowledgements

`POST /api/v1/jobs/{job_id}/readiness-acknowledgements`

Required headers:

- `Idempotency-Key`

请求：

- `warning_code`
- `affected_refs`
- `mapping_version`
- `move_plan_version`
- `message_shown_to_user`

响应：

- ReadinessAcknowledgement
- updated readiness summary

`GET /api/v1/jobs/{job_id}/readiness-acknowledgements`

返回 current/stale acknowledgements。Apply 不需要传 ack ids；后端按当前 job readiness 集合检查是否具备全部 current ack。

### Job creation / scan

如果 UI 支持创建任务：

- `POST /api/v1/jobs`
- `POST /api/v1/jobs/{job_id}/scan`
- both require `Idempotency-Key`

如果第一版不支持，Jobs 页只提供 `scan now` 对现有 watcher，不显示 create job。

### Apply

`POST /api/v1/jobs/{job_id}/apply`

Required headers:

- `Idempotency-Key`

请求：

- `scope`: all_validated
- `validation_id`
- `base_mapping_version`
- `base_move_plan_version`
- `base_move_state_revision`
- `base_review_revision`

服务端必须：

- 校验 validation scope 为 all_for_apply。
- 校验 validation 绑定的 mapping/plan/move_state 与当前一致。
- 校验 readiness acknowledgements current。
- 在 operation start 做 path/current identity recheck。
- 在移动前对完整 destination set 做 reservation。
- 任一 recheck/reservation 阻断时，在移动前整体拒绝。

第一版已支持：

- `scope=all_validated`。
- 只消费 `valid + all_for_apply` validation。
- final validation invalid 时移动前拒绝。
- direct rename，写入 PathEvent、OperationItemResult、DestinationReservation，并在每个 move 前做最终 item recheck。
- 成功后 job 进入 `succeeded`，move items 进入 `moved`。
- apply 运行中 job/move items 进入 `applying`，失败会写失败 response payload 供同 idempotency key 稳定 replay。

第一版暂不支持：

- readiness acknowledgement gate。
- apply_partial 状态机。
- two-stage apply。
- 失败后人工 resolution panel。

第一版响应为同步 `200 OK + ApplyResponse`；后续长任务化时再改为 `202 Accepted + Operation`。

### Retry failed apply items

`POST /api/v1/jobs/{job_id}/apply/retry-failed`

Required headers:

- `Idempotency-Key`

Request:

- SelectionScope limited to retryable failed/proposed items from the partial apply operation
- `base_review_revision`
- `base_move_state_revision`
- validation id for `retry_apply` scope or full all_for_apply validation

Response:

- `202 Accepted`
- Operation resource

### Resolve apply partial

`POST /api/v1/jobs/{job_id}/apply/resolve-partial`

Required headers:

- `Idempotency-Key`

Request:

- `base_review_revision`
- `base_move_state_revision`
- `actions[]`: withdraw_moved_to_review / keep_moved_as_applied / mark_failed_manual_needed / exclude_failed / split_unmoved_to_proposed
- `review_safety_check_token` required iff actions include `withdraw_moved_to_review` or any move-to-review action

Path-changing resolution actions must perform final current path/FileIdentity recheck, create/bind ReviewSafetyLease, acquire DestinationReservation, then write PathEvent before filesystem move.

Response:

- `202 Accepted`
- Operation resource

### Safety API

`GET /api/v1/jobs/{job_id}/safety`

返回：

- roots and generations
- watcher status
- consumer chain digest
- current ReviewExclusionProof summary
- last source gate check summary
- safety token summary/status only; raw `review_safety_check_token` and `source_gate_check_token` are returned only by explicit POST check calls
- active operation-bound lease refs only when an operation is running

`POST /api/v1/jobs/{job_id}/safety/review-proof-check`

Required headers:

- `Idempotency-Key`

Request:

- `base_review_revision`

Response:

- ReviewExclusionProof
- `review_safety_check_token`
- ttl
- covered consumers

`POST /api/v1/jobs/{job_id}/safety/source-gate-check`

Required headers:

- `Idempotency-Key`

Request:

- `gate_kind`: watcher_pause / watch_root_disabled / excluded_target
- `base_review_revision`
- optional external pause lease request fields

Response:

- source gate status
- optional `pause_lease_id`
- `source_gate_check_token`
- ttl
- watcher/root generation

### Withdraw and restore API

`POST /api/v1/jobs/{job_id}/rollback-eligibility`

Required headers:

- `Idempotency-Key`

Request:

- mode: withdraw_to_review / restore_moved_to_source / restore_original_layout_to_source / restore_original_layout_to_review
- SelectionScope
- `base_review_revision`
- `base_move_state_revision`
- optional `review_safety_check_token`
- optional `source_gate_check_token`

Response:

- mode-specific eligibility result
- `check_token`
- `ttl_seconds`
- `checked_move_state_revision`
- `token_status`: current / expired / stale / not_current
- destination preview
- item result summary

`POST /api/v1/jobs/{job_id}/withdraw`

Required headers:

- `Idempotency-Key`

Request:

- mode: withdraw_to_review
- SelectionScope limited to moved items
- `base_review_revision`
- `base_move_state_revision`
- `check_token`
- `review_safety_check_token`

Response:

- 第一版为同步 `200 OK + WithdrawResponse`
- 后续长任务化时再改为 `202 Accepted + Operation`

`POST /api/v1/jobs/{job_id}/restore-moved-to-source`

Required headers:

- `Idempotency-Key`

Request:

- SelectionScope limited to moved/withdrawn/restored items eligible for source restore
- `base_review_revision`
- `base_move_state_revision`
- `check_token`
- `pause_lease_id` or equivalent source gate proof
- `source_gate_check_token`

Response:

- `202 Accepted`
- Operation resource

`POST /api/v1/jobs/{job_id}/restore-original-layout-to-source`

Required headers:

- `Idempotency-Key`

Request:

- SelectionScope over source files or all
- `base_review_revision`
- `base_move_state_revision`
- `check_token`
- `pause_lease_id` or equivalent source gate proof
- `source_gate_check_token`

Response:

- `202 Accepted`
- Operation resource

`POST /api/v1/jobs/{job_id}/restore-original-layout-to-review`

Required headers:

- `Idempotency-Key`

Request:

- SelectionScope over source files or all
- `base_review_revision`
- `base_move_state_revision`
- `check_token`
- `review_safety_check_token`

Response:

- `202 Accepted`
- Operation resource

### LLM failures

`GET /api/v1/jobs/{job_id}/llm-failures?limit=&cursor=`

列表字段：

- id
- stage
- attempt
- tmdb_id
- error_message
- created_at
- preview
- truncated
- char_count

raw detail：

`GET /api/v1/jobs/{job_id}/llm-failures/{failure_id}`

约束：

- detail 支持最大返回大小。
- 超限返回 truncated 标记和下载/分页策略。
- raw reply 中路径和敏感字段按策略脱敏。
- 不生成 plan JSON artifact。
- 不写单独 per-job log 文件；只保留运行时 stdout/stderr/std logging。
- DB 中的 Operation、PathEvent、OperationItemResult、MappingEvidence、LLM failure 是审查和恢复的唯一持久化来源。
- 只持久化失败 LLM raw reply；成功 LLM 回复解析成结构化 mapping/evidence 后丢弃 raw。

## 前端按钮可用性

- `检查所选项`: 有选中 rows 时可用，只做局部检查。
- `全量应用前验证`: 有 proposed move items、readiness 可处理、且没有互斥 operation 时可用。
- `应用全量验证通过的全部项目`: 当前全量 validation 通过、readiness ack 当前、无 blocking errors 时可用。
- `预览重新匹配`: job 不在 path operation 中时可用。
- `确认应用重新匹配`: rematch preview 当前且用户确认时可用。
- `提交修正`: 当前 move items 全部为 proposed，且至少填写一个 correction op 时可用。
- `从 review 重新打开`: 已选择完整 withdrawn episode target 时可用；选择 move item 或部分子项时禁用并说明需要选择完整父行。
- `回撤到 review`: 存在 moved items 时可打开资格检查面板；执行按钮只在当前 check_token + review_safety_check_token 可用时启用，后端 operation start 创建/绑定 ReviewSafetyLease。
- `恢复已移动/已回撤文件到原路径`: 可打开资格检查面板；执行按钮只在当前 check_token + source_gate_check_token 可用时启用，后端 operation start 创建/绑定 SourceGateLease。
- `恢复原目录文件布局到原路径`: archived job、metadata complete 时可打开资格检查面板；执行按钮只在当前 check_token + source_gate_check_token 可用时启用，后端 operation start 创建/绑定 SourceGateLease。
- `恢复原目录文件布局到 review 区`: metadata complete 时可打开资格检查面板；执行按钮只在当前 check_token + review_safety_check_token 可用时启用，后端 operation start 创建/绑定 ReviewSafetyLease。
- `解决部分应用`: job 为 apply_partial 时显示；所有 partial item 都有 resolution action 后才可提交。

按钮禁用时必须显示原因。

## UI 细节

### 路径显示

- 每条路径显示 root badge: source / output / review / archive / fail / temp / missing / unknown。
- basename 作为主显示。
- basename 下方常驻 compact parent path。
- source / current / planned path 在同列内差异化展示。
- 完整路径放 detail panel。
- 支持复制路径。
- 冲突、同名、大小写风险高亮。

### 空状态

- no jobs: scan now / create job if supported
- no TMDB candidates: manual TMDB ID
- no rows match filters: clear filters / show all rows
- focus result stale: refetch focus / clear focus / show all rows
- filtered rows have no available action: explain why and offer clear filters
- no warnings: return to all rows
- no LLM failures: hide raw panel or empty state
- no move items: rescan / source files / LLM failure
- review payload failed: retry / back to Jobs
- operation partial failed: show item results, resolution panel, and safe next actions
- rollback eligibility stale: rerun check

### 窄屏策略

- desktop: 三栏完整工作台。
- tablet 1024px-1279px: source / moves / detail tabs，保留筛选、选择、详情和 sticky actions，只允许非路径变更编辑。
- below 1024px: 禁用编辑/apply/restore/path-changing actions，但保留只读 Match Overview、Item Review、Operation 状态和错误详情，提示需要桌面宽度完成审查和恢复操作。

## 实施阶段

### Phase 0: Review data model and operation contract

- PathRef / FileIdentity。
- DB review snapshot。
- source file records。
- SubtitleLink。
- episode targets。
- MappingAttempt / MappingEvidence / SubtitleLinkEvidence。
- TMDB candidate/episode metadata。
- move item path lifecycle。
- write-ahead path event ledger。
- DestinationReservation。
- job finalize metadata。
- structured review exclusion proof。
- ReviewSafetyLease / SourceGateLease。
- endpoint precondition/effects table。
- status transition tables。
- split version contract。
- ValidationResult。
- ReadinessAcknowledgement。
- Operation resource, polling endpoints, idempotency contract。
- OperationItemResult。
- ProblemDetails。
- OpenAPI / Pydantic schemas。

### Phase 1: Mock UI

- TypeScript types from schema。
- mock review payload。
- Jobs 页面。
- episode target 分组主表。
- 子行 move items。
- readiness panel with drill-down。
- tabs 详情面板。
- Operation Center。
- Settings & Safety。
- persistent filter bar。
- selection model。
- 禁用原因。

### Phase 2: Review API

- `/review`
- `/moves/delta`
- `/operations/{id}`
- `/llm-failures`
- Jobs review summary。

### Phase 3: Rematch/manual correction/reopen

- rematch preview / commit。
- episode target correction。
- SubtitleLink attach/detach/move/variant override。
- exclude / unexclude。
- reopen complete target from review。

### Phase 4: Validate/apply

- selected check。
- all-for-apply validation。
- readiness acknowledgement。
- apply all confirmation。
- apply operation polling。
- move item 状态增量刷新。
- resolve apply partial。

### Phase 5: Withdraw to review

- review exclusion proof。
- unique review path。
- selected/all moved item withdraw。
- eligibility lazy check。
- write-ahead path events。
- item-level result。

### Phase 6: Advanced restore

- restore moved items to source。
- restore original file layout。
- restore original layout into review。
- metadata completeness check。
- identity verification。
- target reservation。
- auto-consumption gate。
- partial/ineligible UX。

### Phase 7: FastAPI static hosting

- Vite build。
- FastAPI static mount。
- README 运行说明。

## 当前落地状态 (2026-07-06)

- 已落地 `/review`、`/operations`、`/safety` 的数据库读模型与静态 review UI。
- 已落地显式触发的 rollback eligibility 检查；GET review 不扫盘、不主动检查 eligibility。
- 已落地 `POST /api/v1/jobs/{job_id}/rematch-preview`：指定 TMDB ID 后重建预览计划，写入 `rematch_preview` operation，返回 episode target、subtitle link、move diff，不替换当前 move rows，不移动文件。
- 已落地 `POST /api/v1/jobs/{job_id}/rematch-commit`：基于已完成 preview operation 和当前 review revision，将未移动的 proposed 计划替换为预览计划，更新 job/review snapshot，写入 `rematch_commit` operation，不移动文件。
- 已落地 `POST /api/v1/jobs/{job_id}/corrections`：对 proposed 计划做人工 episode / subtitle variant 修正，写入 DB plan/snapshot/operation，不移动文件。
- 已落地 `POST /api/v1/jobs/{job_id}/validate`：执行 apply 前基础验证，返回并记录 ValidationResult，不移动文件。
- 已落地 `POST /api/v1/jobs/{job_id}/apply`：消费当前 valid validation，执行 direct rename，记录 PathEvent/OperationItemResult/DestinationReservation。
- 已落地 `POST /api/v1/jobs/{job_id}/withdraw` 第一版：显式 eligibility token 后，将 moved 输出文件回撤到 review bucket，记录 PathEvent/OperationItemResult/DestinationReservation。
- 已落地 `POST /api/v1/jobs/{job_id}/reopen-from-review` 第一版：仅 `all` scope，将 withdrawn move 的 review 文件设为新 source，move 回到 `proposed`，记录 Operation/OperationItemResult，不移动文件。
- 尚未落地 advanced restore 执行、episode-target 粒度 reopen 和完整 safety/source-gate token。

## UI 原则

- 第一屏就是工作台。
- 信息密度优先。
- episode target 分组审查优先。
- 不做 landing page。
- 不允许前端编辑任意 destination path。
- 不把 move 表现成 copy 或 backup。
- 默认撤销是“回撤到 review”，不是“还原到原路径”。
- 完整恢复必须依赖完整记录、身份校验、write-ahead path events 和即时检查。
- 所有失败都要能定位到具体 job/source/target/move item/path event/stage/path。
