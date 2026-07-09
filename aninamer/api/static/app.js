const state = {
  token: localStorage.getItem("aninamer.token") || "",
  jobs: [],
  selectedJobId: null,
  review: null,
  safety: null,
  operations: [],
  reviewSafetyCheck: null,
  rollbackEligibility: null,
  rematchPreview: null,
  applyValidation: null,
  rematchPreviewGeneration: 0,
  detailRequestId: 0,
  detailLoadingJobId: null,
  jobsRefreshing: false,
  tokenExpiryTimer: null,
  tmdbInputSeededJobId: null,
  tmdbInputEditedJobId: null,
  reviewSafetyChecking: false,
  rollbackChecking: false,
  withdrawSubmitting: false,
  reopenSubmitting: false,
  validateChecking: false,
  applySubmitting: false,
  rematchPreviewing: false,
  rematchCommitting: false,
  correctionSubmitting: false,
};

const JOB_AUTO_REFRESH_MS = 30_000;

const UI_LABELS = {
  applied: "已应用",
  blocking: "阻塞",
  blocked: "已阻塞",
  canceled: "已取消",
  chi: "中文",
  chs: "简体",
  cht: "繁体",
  committed_llm_episode_mapping: "已提交 LLM 剧集匹配",
  committed_llm_subtitle_link: "已提交 LLM 字幕关联",
  committed_manual_override: "已提交手动修正",
  committed_rematch: "已提交重匹配",
  committed_rematch_preview: "已提交重匹配预览",
  committed_rematch_subtitle_preview: "已提交重匹配字幕预览",
  current: "当前",
  detected: "自动检测",
  eligible: "可回撤",
  evidence: "证据",
  failed: "失败",
  high: "高",
  ignored: "已忽略",
  ineligible: "不可回撤",
  invalid: "未通过",
  job: "任务",
  llm_episode_mapping: "LLM 剧集匹配",
  llm_subtitle_link: "LLM 字幕关联",
  low: "低",
  manual_override: "手动修正",
  medium: "中",
  missing: "缺失文件",
  moved: "已移动",
  incomplete: "记录不完整",
  identity: "身份不匹配",
  original: "留在原处",
  partial: "部分成功",
  planned: "已计划",
  planned_output: "计划输出",
  preview: "预览",
  preview_only: "仅预览",
  preview_ready: "预览可用",
  proposed: "待处理",
  queued: "排队中",
  ready: "就绪",
  removed: "已移除",
  running: "运行中",
  stale: "已过期",
  succeeded: "成功",
  unknown: "未知",
  valid: "通过",
  verified: "已验证",
  video: "视频",
  warning: "警告",
  withdrawn: "已回撤",
  withdraw_to_review: "回撤到 review",
  restore_moved_to_source: "恢复到源位置",
  subtitle: "字幕",
  changed: "已变更",
  added: "新增",
  unchanged: "未变化",
  summary: "摘要",
  status: "状态",
  blocking: "阻塞",
  warnings: "警告",
  unmapped: "未匹配视频",
  orphanSubs: "未关联字幕",
  tmdb: "TMDB",
  title: "标题",
  originalName: "原名",
  year: "年份",
  candidates: "候选",
  seasons: "季",
  specials: "特别篇",
  episodeCounts: "集数",
  specialEpisodes: "特别篇",
  link: "链接",
  operation: "操作",
  baseRev: "审查版本",
  instructions: "附加提示",
  used: "已使用",
  unused: "未使用",
  watcher: "监控",
  proof: "证明",
  token: "Token",
  validation: "验证",
  checked: "已检查",
  hint: "提示",
  last: "上次检查",
  mode: "模式",
  corrections: "修正",
  review_safety_check: "安全检查",
  rollback_eligibility: "回撤资格",
  rematch_preview: "重匹配预览",
  rematch_subtitle_preview: "重匹配字幕预览",
  rematch_commit: "提交重匹配",
  apply_validation: "应用验证",
  apply: "应用",
  withdraw: "回撤",
  reopen_from_review: "重新打开",
  user: "用户确认",
  "not checked": "未检查",
  "has moved items": "有已移动项目",
  "no moved items": "无已移动项目",
};

const els = {
  runtimeStatus: document.querySelector("#runtimeStatus"),
  authForm: document.querySelector("#authForm"),
  apiToken: document.querySelector("#apiToken"),
  rematchForm: document.querySelector("#rematchForm"),
  tmdbInput: document.querySelector("#tmdbInput"),
  rematchInstructions: document.querySelector("#rematchInstructions"),
  previewRematch: document.querySelector("#previewRematch"),
  commitRematch: document.querySelector("#commitRematch"),
  correctionForm: document.querySelector("#correctionForm"),
  submitCorrection: document.querySelector("#submitCorrection"),
  correctionTargetId: document.querySelector("#correctionTargetId"),
  correctionSeason: document.querySelector("#correctionSeason"),
  correctionStart: document.querySelector("#correctionStart"),
  correctionEnd: document.querySelector("#correctionEnd"),
  correctionSubtitleLinkId: document.querySelector("#correctionSubtitleLinkId"),
  correctionVariant: document.querySelector("#correctionVariant"),
  validateApply: document.querySelector("#validateApply"),
  applyValidated: document.querySelector("#applyValidated"),
  refreshJobs: document.querySelector("#refreshJobs"),
  refreshReview: document.querySelector("#refreshReview"),
  checkReviewSafety: document.querySelector("#checkReviewSafety"),
  checkRollback: document.querySelector("#checkRollback"),
  withdrawChecked: document.querySelector("#withdrawChecked"),
  reopenFromReview: document.querySelector("#reopenFromReview"),
  jobList: document.querySelector("#jobList"),
  jobTitle: document.querySelector("#jobTitle"),
  jobMeta: document.querySelector("#jobMeta"),
  notice: document.querySelector("#notice"),
  readiness: document.querySelector("#readiness"),
  readinessDetailsDisclosure: document.querySelector("#readinessDetailsDisclosure"),
  readinessDetailsSummary: document.querySelector("#readinessDetailsSummary"),
  readinessDetails: document.querySelector("#readinessDetails"),
  tmdbInfo: document.querySelector("#tmdbInfo"),
  tmdbDetails: document.querySelector("#tmdbDetails"),
  tmdbCandidatesDisclosure: document.querySelector("#tmdbCandidatesDisclosure"),
  tmdbCandidatesSummary: document.querySelector("#tmdbCandidatesSummary"),
  tmdbCandidates: document.querySelector("#tmdbCandidates"),
  safetyInfo: document.querySelector("#safetyInfo"),
  validationInfo: document.querySelector("#validationInfo"),
  validationDetailsDisclosure: document.querySelector("#validationDetailsDisclosure"),
  validationDetailsSummary: document.querySelector("#validationDetailsSummary"),
  validationDetails: document.querySelector("#validationDetails"),
  rollbackInfo: document.querySelector("#rollbackInfo"),
  rollbackSafetyStep: document.querySelector("#rollbackSafetyStep"),
  rollbackSafetyStatus: document.querySelector("#rollbackSafetyStatus"),
  rollbackEligibilityStep: document.querySelector("#rollbackEligibilityStep"),
  rollbackEligibilityStatus: document.querySelector("#rollbackEligibilityStatus"),
  rollbackExecuteStep: document.querySelector("#rollbackExecuteStep"),
  rollbackExecuteStatus: document.querySelector("#rollbackExecuteStatus"),
  episodeRows: document.querySelector("#episodeRows"),
  subtitleRows: document.querySelector("#subtitleRows"),
  fileRows: document.querySelector("#fileRows"),
  evidenceList: document.querySelector("#evidenceList"),
  operationRows: document.querySelector("#operationRows"),
  moveRows: document.querySelector("#moveRows"),
  rollbackDetailsDisclosure: document.querySelector("#rollbackDetailsDisclosure"),
  rollbackDetailsSummary: document.querySelector("#rollbackDetailsSummary"),
  rollbackDetails: document.querySelector("#rollbackDetails"),
  rematchInfo: document.querySelector("#rematchInfo"),
  rematchTmdbDetails: document.querySelector("#rematchTmdbDetails"),
  rematchRows: document.querySelector("#rematchRows"),
  confirmDialog: document.querySelector("#confirmDialog"),
  confirmTitle: document.querySelector("#confirmTitle"),
  confirmSummary: document.querySelector("#confirmSummary"),
  confirmDetails: document.querySelector("#confirmDetails"),
  confirmCancel: document.querySelector("#confirmCancel"),
  confirmClose: document.querySelector("#confirmClose"),
  confirmProceed: document.querySelector("#confirmProceed"),
  tabButtonRematch: document.querySelector("#tabButtonRematch"),
};

els.apiToken.value = state.token;

els.authForm.addEventListener("submit", (event) => {
  event.preventDefault();
  state.token = els.apiToken.value.trim();
  localStorage.setItem("aninamer.token", state.token);
  loadRuntimeAndJobs();
});

els.refreshJobs.addEventListener("click", () => loadRuntimeAndJobs());
els.refreshReview.addEventListener("click", () => {
  if (state.selectedJobId) {
    loadJobDetail(state.selectedJobId);
  }
});
els.checkReviewSafety.addEventListener("click", () => checkReviewSafety());
els.checkRollback.addEventListener("click", () => checkRollbackEligibility());
els.withdrawChecked.addEventListener("click", () => withdrawChecked());
els.reopenFromReview.addEventListener("click", () => reopenFromReview());
els.validateApply.addEventListener("click", () => validateApply());
els.applyValidated.addEventListener("click", () => applyValidated());
els.commitRematch.addEventListener("click", () => commitRematch());
els.correctionForm.addEventListener("submit", (event) => {
  event.preventDefault();
  submitCorrection();
});
els.rematchForm.addEventListener("submit", (event) => {
  event.preventDefault();
  previewRematch();
});
els.tmdbInput.addEventListener("input", () => {
  if (state.review) {
    state.tmdbInputEditedJobId = state.review.job.id;
  }
  state.rematchPreview = null;
  state.rematchPreviewGeneration += 1;
  if (state.review) {
    renderReview();
  }
});
els.rematchInstructions.addEventListener("input", () => {
  state.rematchPreview = null;
  state.rematchPreviewGeneration += 1;
  if (state.review) {
    renderReview();
  }
});

const tabButtons = Array.from(document.querySelectorAll(".tab"));

tabButtons.forEach((button) => {
  button.addEventListener("click", () => activateTab(button));
  button.addEventListener("keydown", (event) => handleTabKeydown(event, button));
});

function activateTab(button, options = {}) {
  tabButtons.forEach((item) => {
    const selected = item === button;
    const panel = document.querySelector(`#tab-${item.dataset.tab}`);
    item.classList.toggle("active", selected);
    item.setAttribute("aria-selected", selected ? "true" : "false");
    item.tabIndex = selected ? 0 : -1;
    panel?.classList.toggle("active", selected);
  });
  if (options.focus) {
    button.focus();
  }
}

function handleTabKeydown(event, button) {
  const currentIndex = tabButtons.indexOf(button);
  if (currentIndex < 0) {
    return;
  }
  let nextIndex = null;
  if (event.key === "ArrowRight") {
    nextIndex = (currentIndex + 1) % tabButtons.length;
  } else if (event.key === "ArrowLeft") {
    nextIndex = (currentIndex - 1 + tabButtons.length) % tabButtons.length;
  } else if (event.key === "Home") {
    nextIndex = 0;
  } else if (event.key === "End") {
    nextIndex = tabButtons.length - 1;
  }
  if (nextIndex === null) {
    return;
  }
  event.preventDefault();
  activateTab(tabButtons[nextIndex], { focus: true });
}

async function api(path, options = {}) {
  const headers = new Headers(options.headers || {});
  if (state.token) {
    headers.set("Authorization", `Bearer ${state.token}`);
  }
  if (options.body && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  const response = await fetch(path, { ...options, headers });
  const text = await response.text();
  const payload = text ? JSON.parse(text) : null;
  if (!response.ok) {
    const message = payload?.error?.message || response.statusText;
    const error = new Error(message);
    error.payload = payload;
    error.status = response.status;
    throw error;
  }
  return payload;
}

async function loadRuntimeAndJobs(options = {}) {
  const refreshSelectedDetail = options.refreshSelectedDetail ?? true;
  const quiet = Boolean(options.quiet);
  if (state.jobsRefreshing) {
    return;
  }
  state.jobsRefreshing = true;
  if (!quiet) {
    clearNotice();
  }
  try {
    const [runtime, jobs] = await Promise.all([
      api("/api/v1/runtime"),
      api("/api/v1/jobs"),
    ]);
    els.runtimeStatus.textContent = `${runtime.watch_root_keys.length} roots`;
    state.jobs = jobs.items;
    renderJobs();
    if (state.selectedJobId && state.jobs.some((job) => job.id === state.selectedJobId)) {
      if (refreshSelectedDetail) {
        await loadJobDetail(state.selectedJobId);
      }
    } else if (state.selectedJobId) {
      state.selectedJobId = null;
      state.rollbackEligibility = null;
      state.rematchPreview = null;
      state.applyValidation = null;
      clearDetail();
      renderJobs();
    }
  } catch (error) {
    els.runtimeStatus.textContent = "连接失败";
    if (!quiet) {
      showNotice(error.message, true);
    }
  } finally {
    state.jobsRefreshing = false;
  }
}

function renderJobs() {
  els.jobList.replaceChildren();
  if (!state.jobs.length) {
    const node = empty("暂无 job，等待扫描");
    node.classList.add("job-empty");
    els.jobList.append(node);
    return;
  }
  for (const job of state.jobs) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `job-item${job.id === state.selectedJobId ? " active" : ""}`;
    button.innerHTML = `
      <strong>${escapeHtml(job.series_name)}</strong>
      <span>${job.id} · ${escapeHtml(labelFor(job.status))} · 视频 ${job.video_moves_count} · 字幕 ${job.subtitle_moves_count}</span>
    `;
    button.addEventListener("click", () => loadJobDetail(job.id));
    els.jobList.append(button);
  }
}

async function loadJobDetail(jobId, options = {}) {
  const requestId = state.detailRequestId + 1;
  state.detailRequestId = requestId;
  const previousJobId = state.selectedJobId;
  const keepCurrentDetail = previousJobId === jobId && state.review?.job.id === jobId;
  if (!options.preserveNotice) {
    clearNotice();
  }
  state.selectedJobId = jobId;
  state.detailLoadingJobId = jobId;
  if (!options.preserveRollback) {
    state.rollbackEligibility = null;
  }
  if (!options.preserveReviewSafety) {
    state.reviewSafetyCheck = null;
  }
  if (!options.preserveRematch) {
    state.rematchPreview = null;
    state.rematchPreviewGeneration += 1;
  }
  if (!options.preserveValidation) {
    state.applyValidation = null;
  }
  if (keepCurrentDetail) {
    renderReview();
  } else {
    clearDetail({ preserveTmdbInput: previousJobId === jobId });
    state.detailLoadingJobId = jobId;
    els.jobTitle.textContent = "加载中";
    els.jobMeta.textContent = `job ${jobId}`;
  }
  renderJobs();
  els.refreshReview.disabled = true;
  try {
    const [review, safety, operations] = await Promise.all([
      api(`/api/v1/jobs/${jobId}/review`),
      api(`/api/v1/jobs/${jobId}/safety`),
      api(`/api/v1/jobs/${jobId}/operations`),
    ]);
    if (state.detailRequestId !== requestId || state.selectedJobId !== jobId) {
      return;
    }
    state.detailLoadingJobId = null;
    state.review = review;
    state.safety = safety;
    state.operations = operations.items;
    renderReview();
  } catch (error) {
    if (state.detailRequestId !== requestId || state.selectedJobId !== jobId) {
      return;
    }
    state.detailLoadingJobId = null;
    if (keepCurrentDetail) {
      renderReview();
    } else {
      clearDetail();
    }
    showNotice(error.message, true);
  }
}

function renderReview() {
  const { review, safety } = state;
  if (!review) {
    return;
  }
  els.jobTitle.textContent = review.job.series_name;
  const loading = state.detailLoadingJobId === review.job.id;
  els.jobMeta.textContent = [
    `job ${review.job.id}`,
    labelFor(review.job.status),
    `审查 ${review.review_revision}`,
    `匹配 ${review.versions.mapping_version}`,
    `计划 ${review.versions.move_plan_version}`,
    `状态 ${review.versions.move_state_revision}`,
    loading ? "刷新中" : "",
  ].filter(Boolean).join(" · ");
  els.refreshReview.disabled = loading;
  const actionBusy = isActionBusy();
  const safetyForJob =
    state.reviewSafetyCheck?.jobId === review.job.id
      ? state.reviewSafetyCheck.result
      : null;
  const reviewSafetyReady =
    safetyForJob &&
    safetyForJob.status === "verified" &&
    isTimedTokenCurrent(safetyForJob, "review_safety_check_token");
  const checkSafetyReason = actionBusy ? "等待当前操作完成后再检查安全" : "";
  setButtonState(els.checkReviewSafety, Boolean(checkSafetyReason), checkSafetyReason);
  const checkRollbackReason =
    state.rollbackChecking
      ? "正在检查回撤资格"
      : actionBusy
        ? "等待当前操作完成后再检查回撤"
        : !reviewSafetyReady
          ? "需先通过安全检查"
          : !review.rollback_capability_hints.has_moved_items
            ? "没有已移动项目可检查回撤"
            : "";
  setButtonState(
    els.checkRollback,
    Boolean(checkRollbackReason),
    checkRollbackReason,
  );
  const proposedMoveCount = review.move_items.filter(
    (item) => item.status === "proposed",
  ).length;
  const validateReason =
    state.validateChecking
      ? "正在验证"
      : actionBusy
        ? "等待当前操作完成后再验证"
        : !proposedMoveCount
          ? "没有待处理移动项可验证"
          : "";
  setButtonState(els.validateApply, Boolean(validateReason), validateReason);
  const previewReason =
    actionBusy ? "等待当前操作完成后再预览重匹配" : "";
  setButtonState(els.previewRematch, Boolean(previewReason), previewReason);
  const previewForJob =
    state.rematchPreview?.jobId === review.job.id ? state.rematchPreview.result : null;
  const preferredTmdbId =
    previewForJob?.tmdb_id || review.selected_tmdb_info.selected_tmdb_id;
  if (
    state.tmdbInputSeededJobId !== review.job.id &&
    state.tmdbInputEditedJobId !== review.job.id &&
    preferredTmdbId
  ) {
    els.tmdbInput.value = preferredTmdbId;
    state.tmdbInputSeededJobId = review.job.id;
  }
  renderDefinitionList(els.readiness, {
    status: review.readiness_summary.status || "unknown",
    blocking: review.readiness_summary.blocking_count,
    warning: review.readiness_summary.warning_count,
    unmapped: review.readiness_summary.unmapped_video_count,
    orphanSubs: review.readiness_summary.unassociated_subtitle_count,
  });
  renderReadinessDetails(review.readiness_summary);
  updateDisclosureSummary(
    els.readinessDetailsDisclosure,
    els.readinessDetailsSummary,
    "就绪明细",
    readinessDisclosureText(review.readiness_summary),
    hasReadinessIssues(review.readiness_summary),
  );
  renderTmdbInfo(review.selected_tmdb_info);
  renderTmdbCandidates(review.selected_tmdb_info);
  updateDisclosureSummary(
    els.tmdbCandidatesDisclosure,
    els.tmdbCandidatesSummary,
    "TMDB 候选",
    `${review.selected_tmdb_info.candidates.length} 个`,
    review.selected_tmdb_info.candidates.length > 0,
  );
  renderDefinitionList(els.safetyInfo, {
    watcher: safety?.watcher_status?.status || "-",
    proof: safety?.review_exclusion_proof?.status || review.review_exclusion_proof.status,
    token: tokenLabel(safetyForJob, reviewSafetyReady),
  });
  const rawValidationForJob =
    state.applyValidation?.jobId === review.job.id ? state.applyValidation.result : null;
  const validationForJob =
    rawValidationForJob && validationMatchesReview(rawValidationForJob, review)
      ? rawValidationForJob
      : null;
  renderDefinitionList(els.validationInfo, {
    status: validationForJob?.status || "not checked",
    validation: validationForJob?.validation_id || "-",
    checked: validationForJob?.item_result_summary?.checked || 0,
    blocking: validationForJob?.blocking_items?.length || 0,
    warnings: validationForJob?.warning_items?.length || 0,
  });
  renderValidationDetails(validationForJob);
  updateDisclosureSummary(
    els.validationDetailsDisclosure,
    els.validationDetailsSummary,
    "验证明细",
    validationDisclosureText(validationForJob),
    hasValidationIssues(validationForJob),
  );
  const applyReason =
    actionBusy
      ? "等待当前操作完成后再应用"
      : state.validateChecking
        ? "等待验证完成"
        : !validationForJob
          ? "需先运行全量验证"
          : validationForJob.status !== "valid"
            ? "验证未通过"
            : "";
  setButtonState(els.applyValidated, Boolean(applyReason), applyReason);
  const rollbackForJob =
    state.rollbackEligibility?.jobId === review.job.id
      ? state.rollbackEligibility.result
      : null;
  const rollbackReady =
    rollbackForJob &&
    rollbackForJob.mode === "withdraw_to_review" &&
    rollbackForJob.status === "eligible" &&
    isTimedTokenCurrent(rollbackForJob, "check_token");
  const withdrawReason =
    state.rollbackChecking
      ? "等待回撤资格检查完成"
      : actionBusy
        ? "等待当前操作完成后再执行回撤"
        : !reviewSafetyReady
          ? "需先通过安全检查"
          : !rollbackReady
            ? "需先通过回撤资格检查"
            : "";
  setButtonState(els.withdrawChecked, Boolean(withdrawReason), withdrawReason);
  const reopenReason =
    state.rollbackChecking
      ? "等待回撤资格检查完成"
      : actionBusy
        ? "等待当前操作完成后再重新打开"
        : !review.rollback_capability_hints.has_review_records
          ? "没有 review 记录可重新打开"
          : !reviewSafetyReady
            ? "需先通过安全检查"
            : "";
  setButtonState(els.reopenFromReview, Boolean(reopenReason), reopenReason);
  renderDefinitionList(els.rollbackInfo, {
    hint: review.rollback_capability_hints.has_moved_items ? "has moved items" : "no moved items",
    last: review.last_rollback_check_summary.status,
    mode: review.last_rollback_check_summary.mode || "-",
    token: tokenLabel(rollbackForJob, rollbackReady),
  });
  scheduleTokenExpiryRender([safetyForJob, rollbackForJob]);
  renderRollbackFlow({
    reviewSafetyReady,
    rollbackReady,
    rollbackForJob,
    canReopen: review.rollback_capability_hints.has_review_records,
    canWithdraw: review.rollback_capability_hints.has_moved_items,
    checkRollbackReason,
    withdrawReason,
    reopenReason,
  });
  renderEpisodeRows(review.episode_targets);
  renderSubtitleRows(review.subtitle_links);
  renderFileRows(review.source_files);
  renderMoveRows(review.move_items);
  renderEvidence(review.mapping_evidence);
  renderOperations(state.operations);
  renderRollbackDetails(
    rollbackForJob
  );
  renderRematchPreview(
    previewForJob
  );
  const commitRematchReason =
    actionBusy
      ? "等待当前操作完成后再提交预览"
      : !previewForJob
        ? "需先生成重匹配预览"
        : "";
  setButtonState(
    els.commitRematch,
    Boolean(commitRematchReason),
    commitRematchReason,
  );
  const correctionReason =
    actionBusy
      ? "等待当前操作完成后再提交修正"
      : !review.move_items.every((item) => item.status === "proposed")
        ? "只有待处理计划可手动修正"
        : "";
  setButtonState(els.submitCorrection, Boolean(correctionReason), correctionReason);
}

function isWriteBusy() {
  return (
    state.applySubmitting ||
    state.withdrawSubmitting ||
    state.reopenSubmitting ||
    state.rematchCommitting ||
    state.correctionSubmitting
  );
}

function isActionBusy() {
  return (
    isWriteBusy() ||
    state.validateChecking ||
    state.rematchPreviewing ||
    state.reviewSafetyChecking ||
    state.rollbackChecking ||
    state.detailLoadingJobId !== null
  );
}

function labelFor(value) {
  if (value === null || value === undefined || value === "") {
    return "-";
  }
  const text = String(value);
  if (UI_LABELS[text]) {
    return UI_LABELS[text];
  }
  if (text.includes(" · ")) {
    return text
      .split(" · ")
      .map((part) => UI_LABELS[part] || part)
      .join(" · ");
  }
  return text;
}

function ensureDetailIdle(message) {
  if (state.detailLoadingJobId === null) {
    return true;
  }
  showNotice(message, true);
  renderReview();
  return false;
}

function startJobsAutoRefresh() {
  window.setInterval(refreshJobsQuietly, JOB_AUTO_REFRESH_MS);
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden) {
      refreshJobsQuietly();
    }
  });
}

function refreshJobsQuietly() {
  if (!state.token || document.hidden || isActionBusy() || state.jobsRefreshing) {
    return;
  }
  loadRuntimeAndJobs({ refreshSelectedDetail: false, quiet: true });
}

function isTimedTokenCurrent(result, tokenField) {
  const remaining = tokenRemainingMs(result);
  return Boolean(
    result &&
      result.token_status === "current" &&
      result[tokenField] &&
      remaining !== null &&
      remaining > 0
  );
}

function tokenLabel(result, ready) {
  if (!result) {
    return "-";
  }
  const status =
    ready
      ? "current"
      : result.token_status === "current" && tokenExpired(result)
        ? "stale"
        : result.token_status || "-";
  const expiry = tokenExpiryDate(result);
  if (!expiry) {
    return labelFor(status);
  }
  return `${labelFor(status)} · 到期 ${expiry.toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  })}`;
}

function tokenExpired(result) {
  const remaining = tokenRemainingMs(result);
  return remaining !== null && remaining <= 0;
}

function tokenRemainingMs(result) {
  if (
    !result?.checked_at ||
    result.ttl_seconds === null ||
    result.ttl_seconds === undefined ||
    !Number.isFinite(Number(result.ttl_seconds))
  ) {
    return null;
  }
  const checkedAt = Date.parse(result.checked_at);
  if (!Number.isFinite(checkedAt)) {
    return null;
  }
  return checkedAt + Number(result.ttl_seconds) * 1000 - Date.now();
}

function tokenExpiryDate(result) {
  const remaining = tokenRemainingMs(result);
  if (remaining === null) {
    return null;
  }
  return new Date(Date.now() + remaining);
}

function scheduleTokenExpiryRender(results) {
  if (state.tokenExpiryTimer) {
    clearTimeout(state.tokenExpiryTimer);
    state.tokenExpiryTimer = null;
  }
  const delays = results
    .map(tokenRemainingMs)
    .filter((remaining) => remaining !== null && remaining > 0);
  if (!delays.length) {
    return;
  }
  state.tokenExpiryTimer = setTimeout(() => {
    state.tokenExpiryTimer = null;
    if (state.review) {
      renderReview();
    }
  }, Math.min(...delays) + 1000);
}

function renderRollbackFlow({
  reviewSafetyReady,
  rollbackReady,
  rollbackForJob,
  canReopen,
  canWithdraw,
  checkRollbackReason,
  withdrawReason,
  reopenReason,
}) {
  const safetyMode = reviewSafetyReady
    ? "done"
    : state.reviewSafetyChecking
      ? "active"
      : "pending";
  const safetyText = reviewSafetyReady
    ? "已通过"
    : state.reviewSafetyChecking
      ? "检查中"
      : "未检查";
  setRollbackStep(els.rollbackSafetyStep, els.rollbackSafetyStatus, safetyText, safetyMode);

  const eligibilityMode = rollbackReady
    ? "done"
    : state.rollbackChecking
      ? "active"
      : !canWithdraw
        ? "blocked"
        : checkRollbackReason
          ? "pending"
          : "pending";
  const eligibilityText = rollbackReady
    ? "已通过"
    : state.rollbackChecking
      ? "检查中"
      : !canWithdraw
        ? "无已移动项目"
        : rollbackForJob
          ? `${labelFor(rollbackForJob.status)} · ${labelFor(rollbackForJob.token_status)}`
          : checkRollbackReason || "未检查";
  setRollbackStep(
    els.rollbackEligibilityStep,
    els.rollbackEligibilityStatus,
    eligibilityText,
    eligibilityMode,
  );

  const canExecuteWithdraw = !withdrawReason;
  const canExecuteReopen = !reopenReason;
  const executeText =
    state.withdrawSubmitting || state.reopenSubmitting
      ? "执行中"
      : canExecuteWithdraw && canExecuteReopen
        ? "可回撤 / 可重新打开"
        : canExecuteWithdraw
          ? "可执行回撤"
          : canExecuteReopen
            ? "可重新打开"
            : withdrawReason || reopenReason || "等待前置条件";
  const executeMode =
    state.withdrawSubmitting || state.reopenSubmitting
      ? "active"
      : canExecuteWithdraw || canExecuteReopen
        ? "done"
        : canReopen || canWithdraw
          ? "pending"
          : "blocked";
  setRollbackStep(
    els.rollbackExecuteStep,
    els.rollbackExecuteStatus,
    executeText,
    executeMode,
  );
}

function setRollbackStep(step, status, text, mode) {
  step.className = `rollback-step ${mode}`;
  status.textContent = text;
}

function setButtonState(button, disabled, reason = "") {
  button.disabled = disabled;
  button.title = disabled && reason ? reason : "";
}

function updateDisclosureSummary(details, summary, label, status, open) {
  if (!details || !summary) {
    return;
  }
  const labelNode = document.createElement("span");
  labelNode.className = "disclosure-label";
  labelNode.textContent = label;
  const statusNode = document.createElement("span");
  statusNode.className = "disclosure-status";
  statusNode.textContent = status;
  summary.replaceChildren(labelNode, statusNode);
  details.open = Boolean(open);
}

function readinessDisclosureText(summary) {
  const blocking = summary?.blocking_count || 0;
  const warning = summary?.warning_count || 0;
  if (!blocking && !warning) {
    return "无阻塞/警告";
  }
  return `${blocking} 阻塞 / ${warning} 警告`;
}

function hasReadinessIssues(summary) {
  return Boolean((summary?.blocking_count || 0) || (summary?.warning_count || 0));
}

function validationDisclosureText(validation) {
  if (!validation) {
    return "尚未运行";
  }
  const blocking = validation.blocking_items?.length || 0;
  const warning = validation.warning_items?.length || 0;
  if (!blocking && !warning) {
    return "无阻塞/警告";
  }
  return `${blocking} 阻塞 / ${warning} 警告`;
}

function hasValidationIssues(validation) {
  return Boolean(
    (validation?.blocking_items?.length || 0) ||
      (validation?.warning_items?.length || 0)
  );
}

function rollbackDisclosureText(result) {
  if (!result) {
    return "尚未检查";
  }
  const previewCount = result.destination_preview?.length || 0;
  const problemCount = rollbackProblemCount(result);
  return `${previewCount} 项预览 / ${problemCount} 异常`;
}

function rollbackProblemCount(result) {
  return [
    result?.missing_files,
    result?.incomplete_records,
    result?.identity_mismatches,
    result?.target_conflicts,
  ].reduce((count, items) => count + (Array.isArray(items) ? items.length : 0), 0);
}

function renderTmdbInfo(info) {
  els.tmdbInfo.replaceChildren();
  appendDefinitionItem(els.tmdbInfo, "tmdb", tmdbLinkCell(info?.selected_tmdb_id));
  appendDefinitionItem(els.tmdbInfo, "title", info?.title_zh_cn || "-");
  appendDefinitionItem(els.tmdbInfo, "originalName", info?.original_name || "-");
  appendDefinitionItem(els.tmdbInfo, "year", info?.year || "-");
  appendDefinitionItem(els.tmdbInfo, "candidates", info?.candidates?.length || 0);
  appendDefinitionItem(els.tmdbInfo, "episodeCounts", formatSeasonCounts(info?.seasons));
  renderTmdbSeasonDetails(els.tmdbDetails, info);
}

function renderTmdbSeasonDetails(target, info) {
  target.replaceChildren();
  const seasons = sortedSeasons(info?.seasons);
  if (seasons.length) {
    target.append(seasonChipList(seasons));
  }
  if (info?.specials?.length) {
    target.append(specialsDisclosure(info.specials));
  }
}

function renderSelectedTmdbBlock(target, info) {
  target.replaceChildren();
  if (!info?.selected_tmdb_id) {
    target.append(empty("生成预览后显示 TMDB 明细"));
    return;
  }
  const block = document.createElement("section");
  block.className = "tmdb-selected-block";
  const header = document.createElement("header");
  const title = document.createElement("strong");
  title.textContent = info.title_zh_cn || `TMDB ${info.selected_tmdb_id}`;
  header.append(title, tmdbAnchor(info.selected_tmdb_id, "打开 TMDB"));
  const meta = document.createElement("div");
  meta.className = "tmdb-selected-meta";
  meta.textContent = [
    info.original_name,
    info.year ? `${info.year}` : "",
    formatSeasonCounts(info.seasons),
  ].filter(Boolean).join(" · ");
  block.append(header, meta);
  const seasons = sortedSeasons(info.seasons);
  if (seasons.length) {
    block.append(seasonChipList(seasons));
  }
  if (info.specials?.length) {
    block.append(specialsDisclosure(info.specials));
  }
  target.append(block);
}

function tmdbLinkCell(tmdbId) {
  if (!tmdbId) {
    return "-";
  }
  const wrap = document.createElement("span");
  wrap.className = "tmdb-link-cell";
  const idText = document.createElement("span");
  idText.textContent = String(tmdbId);
  wrap.append(idText, tmdbAnchor(tmdbId, "打开"));
  return wrap;
}

function tmdbAnchor(tmdbId, text) {
  const link = document.createElement("a");
  link.href = tmdbTvUrl(tmdbId);
  link.target = "_blank";
  link.rel = "noopener noreferrer";
  link.textContent = text;
  return link;
}

function textSpan(text) {
  const span = document.createElement("span");
  span.textContent = text;
  return span;
}

function tmdbTvUrl(tmdbId) {
  return `https://www.themoviedb.org/tv/${encodeURIComponent(String(tmdbId))}`;
}

function formatSeasonCounts(seasons) {
  const labels = sortedSeasons(seasons).map(seasonCountLabel);
  return labels.join(" / ") || "-";
}

function sortedSeasons(seasons) {
  if (!Array.isArray(seasons)) {
    return [];
  }
  return seasons
    .filter((season) => Number.isInteger(Number(season.season_number)))
    .slice()
    .sort((left, right) => Number(left.season_number) - Number(right.season_number));
}

function seasonCountLabel(season) {
  const seasonNumber = Number(season.season_number);
  const count = Number(season.episode_count || 0);
  return `S${pad2(seasonNumber)} ${count}集`;
}

function seasonChipList(seasons) {
  const list = document.createElement("div");
  list.className = "season-chip-list";
  for (const season of seasons) {
    const chip = document.createElement("span");
    chip.className = "season-chip";
    chip.textContent = seasonCountLabel(season);
    list.append(chip);
  }
  return list;
}

function specialsDisclosure(specials) {
  const details = document.createElement("details");
  details.className = "tmdb-specials disclosure-details";
  const summary = document.createElement("summary");
  const label = document.createElement("span");
  label.className = "disclosure-label";
  label.textContent = "特别篇";
  const status = document.createElement("span");
  status.className = "disclosure-status";
  status.textContent = `${specials.length} 个`;
  summary.append(label, status);
  const list = document.createElement("div");
  list.className = "tmdb-special-list";
  for (const item of specials.slice(0, 20)) {
    const rowNode = document.createElement("div");
    rowNode.className = "tmdb-special-row";
    rowNode.textContent = [
      `S00E${pad2(item.episode_number)}`,
      item.name,
      item.air_date,
    ].filter(Boolean).join(" · ");
    list.append(rowNode);
  }
  if (specials.length > 20) {
    list.append(empty(`还有 ${specials.length - 20} 个特别篇未显示`));
  }
  details.append(summary, list);
  return details;
}

function renderReadinessDetails(summary) {
  els.readinessDetails.replaceChildren();
  const refs = summary?.affected_refs || {};
  const groups = [
    {
      key: "unmapped_videos",
      label: "未匹配视频",
      severity: "blocking",
    },
    {
      key: "unassociated_subtitles",
      label: "未关联字幕",
      severity: "warning",
    },
    {
      key: "ignored_files",
      label: "忽略文件",
      severity: "warning",
    },
  ];
  let rendered = 0;
  for (const group of groups) {
    for (const item of refs[group.key] || []) {
      rendered += 1;
      appendDetailItem(els.readinessDetails, {
        label: group.label,
        meta: group.severity,
        fields: compactFields([
          ["源文件", item.source_file_id],
          ["路径", optionalPath(item.path)],
          ["原因", item.ignored_reason],
        ]),
      });
    }
  }
  if (!rendered) {
    els.readinessDetails.append(empty("没有阻塞/警告项"));
  }
}

function renderValidationDetails(validation) {
  els.validationDetails.replaceChildren();
  if (!validation) {
    els.validationDetails.append(empty("尚未运行验证"));
    return;
  }
  let rendered = 0;
  for (const item of validation.blocking_items || []) {
    rendered += 1;
    appendValidationProblem("blocking", item);
  }
  for (const item of validation.warning_items || []) {
    rendered += 1;
    appendValidationProblem("warning", item);
  }
  if (!rendered) {
    els.validationDetails.append(empty("验证未发现阻塞/警告项"));
  }
}

function appendValidationProblem(severity, item) {
  appendDetailItem(els.validationDetails, {
    label: item.code || "验证问题",
    meta: severity,
    fields: compactFields([
      ["消息", item.message],
      ["移动项", item.move_item_id],
      ["源 ID", item.source_id],
      ["类型", labelFor(item.kind)],
      ["移动状态", labelFor(item.move_status)],
      ["来源", optionalPath(item.source)],
      ["目标", optionalPath(item.destination)],
      ["输出根目录", optionalPath(item.output_root)],
      ["父目录", optionalPath(item.parent)],
      ["冲突项", formatList(item.conflicting_move_item_ids)],
    ]),
  });
}

function appendDetailItem(target, item) {
  const node = document.createElement("article");
  node.className = "detail-item";
  const header = document.createElement("header");
  const title = document.createElement("strong");
  title.textContent = item.label || "-";
  const meta = document.createElement("span");
  meta.className = "muted";
  const rawMeta = item.meta || "";
  const metaLabel = rawMeta ? labelFor(rawMeta) : "";
  meta.textContent = metaLabel;
  if (rawMeta && metaLabel !== String(rawMeta)) {
    meta.title = String(rawMeta);
  }
  header.append(title, meta);
  node.append(header);
  if (item.fields?.length) {
    const list = document.createElement("dl");
    for (const [label, value] of item.fields) {
      const dt = document.createElement("dt");
      dt.textContent = label;
      const dd = document.createElement("dd");
      dd.textContent = value ?? "-";
      list.append(dt, dd);
    }
    node.append(list);
  }
  target.append(node);
}

function compactFields(fields) {
  return fields.filter(([, value]) => value !== null && value !== undefined && value !== "");
}

function formatList(value) {
  if (!Array.isArray(value) || !value.length) {
    return "";
  }
  return value.join(", ");
}

function yearFromDate(value) {
  if (typeof value !== "string" || value.length < 4) {
    return "";
  }
  return value.slice(0, 4);
}

function optionalPath(value) {
  if (value === null || value === undefined || value === "") {
    return "";
  }
  return displayPath(value);
}

function renderTmdbCandidates(info) {
  els.tmdbCandidates.replaceChildren();
  const candidates = info?.candidates || [];
  if (!candidates.length) {
    els.tmdbCandidates.append(empty("没有 TMDB 候选"));
    return;
  }
  const currentValue = Number(els.tmdbInput.value);
  for (const candidate of candidates) {
    const item = document.createElement("div");
    item.className = `tmdb-candidate${candidate.id === currentValue ? " selected" : ""}`;
    const button = document.createElement("button");
    button.type = "button";
    button.className = "tmdb-candidate-select";
    button.title = `填入 TMDB ID ${candidate.id}`;
    const title = candidate.name || candidate.original_name || `TMDB ${candidate.id}`;
    const year = candidate.year || yearFromDate(candidate.first_air_date) || "-";
    const countries = Array.isArray(candidate.origin_country)
      ? candidate.origin_country.join(", ")
      : "";
    const popularity = candidate.popularity ?? "-";
    const votes = candidate.vote_count ?? "-";
    button.innerHTML = `
      <strong>${escapeHtml(title)}</strong>
      <span>${escapeHtml(candidate.id)} · ${escapeHtml(year)}${countries ? ` · ${escapeHtml(countries)}` : ""}</span>
      <span>${escapeHtml(candidate.original_name || "")}</span>
    `;
    button.addEventListener("click", () => selectTmdbCandidate(candidate.id));
    const meta = document.createElement("div");
    meta.className = "tmdb-candidate-meta";
    meta.append(
      textSpan(`热度 ${popularity}`),
      textSpan(`投票 ${votes}`),
      tmdbAnchor(candidate.id, "TMDB"),
    );
    item.append(button, meta);
    els.tmdbCandidates.append(item);
  }
}

function selectTmdbCandidate(tmdbId) {
  if (!state.review || !Number.isInteger(Number(tmdbId)) || Number(tmdbId) <= 0) {
    return;
  }
  els.tmdbInput.value = String(tmdbId);
  state.tmdbInputEditedJobId = state.review.job.id;
  state.rematchPreview = null;
  state.rematchPreviewGeneration += 1;
  renderReview();
}

function renderEpisodeRows(items) {
  els.episodeRows.replaceChildren();
  if (!items.length) {
    els.episodeRows.append(emptyTableRow(5, "暂无 episode target"));
    return;
  }
  for (const item of items) {
    const tr = row([
      item.id,
      item.video_source_id,
      formatEpisode(item),
      pill(item.status),
      item.subtitle_link_ids.join(", ") || "-",
    ]);
    makeClickableRow(tr, "点击填入 episode 修正", () => prefillEpisodeCorrection(item));
    els.episodeRows.append(tr);
  }
}

function renderSubtitleRows(items) {
  els.subtitleRows.replaceChildren();
  if (!items.length) {
    els.subtitleRows.append(emptyTableRow(5, "暂无字幕关联"));
    return;
  }
  for (const item of items) {
    const tr = row([
      item.id,
      item.episode_target_id,
      item.subtitle_source_id,
      item.variant_override || item.variant_detected,
      pill(item.status),
    ]);
    makeClickableRow(tr, "点击填入字幕修正", () => prefillSubtitleCorrection(item));
    els.subtitleRows.append(tr);
  }
}

function makeClickableRow(tr, title, callback) {
  tr.classList.add("clickable-row");
  tr.tabIndex = 0;
  tr.title = title;
  tr.addEventListener("click", callback);
  tr.addEventListener("keydown", (event) => {
    if (event.key !== "Enter" && event.key !== " ") {
      return;
    }
    event.preventDefault();
    callback();
  });
}

function prefillEpisodeCorrection(item) {
  els.correctionTargetId.value = item.id;
  els.correctionSeason.value = item.season;
  els.correctionStart.value = item.episode_start;
  els.correctionEnd.value = item.episode_end;
  els.correctionSubtitleLinkId.value = "";
  els.correctionVariant.value = "detected";
}

function prefillSubtitleCorrection(item) {
  els.correctionTargetId.value = "";
  els.correctionSeason.value = "";
  els.correctionStart.value = "";
  els.correctionEnd.value = "";
  els.correctionSubtitleLinkId.value = item.id;
  const variant = item.variant_override || item.variant_detected;
  els.correctionVariant.value = ["chs", "cht", "chi"].includes(variant)
    ? variant
    : "detected";
}

function renderFileRows(items) {
  els.fileRows.replaceChildren();
  if (!items.length) {
    els.fileRows.append(emptyTableRow(6, "暂无源文件"));
    return;
  }
  for (const item of items) {
    els.fileRows.append(row([
      item.id,
      labelFor(item.kind),
      pathCell(item.original_path),
      pathCell(item.current_path),
      pill(item.final_disposition, item.final_disposition === "failed" ? "error" : ""),
      item.ignored_reason || item.detected_subtitle_variant || "-",
    ]));
  }
}

function renderMoveRows(items) {
  els.moveRows.replaceChildren();
  if (!items.length) {
    els.moveRows.append(emptyTableRow(7, "暂无移动项"));
    return;
  }
  for (const item of items) {
    els.moveRows.append(row([
      item.id,
      item.source_id,
      labelFor(item.kind),
      pathCell(item.source_path),
      pathCell(item.planned_destination_path),
      pathCell(item.current_path),
      pill(item.status, item.status === "failed" ? "error" : ""),
    ]));
  }
}

function renderEvidence(items) {
  els.evidenceList.replaceChildren();
  if (!items.length) {
    els.evidenceList.append(empty("暂无证据"));
    return;
  }
  for (const item of items) {
    const node = document.createElement("article");
    node.className = "evidence-item";
    const header = document.createElement("header");
    const title = document.createElement("strong");
    title.textContent = labelFor(item.evidence_kind || "evidence");
    header.append(title, pill(item.confidence || "unknown"));
    const summary = document.createElement("div");
    summary.textContent = item.summary || "";
    node.append(header, summary);
    els.evidenceList.append(node);
  }
}

function renderOperations(items) {
  els.operationRows.replaceChildren();
  if (!items.length) {
    els.operationRows.append(emptyTableRow(5, "暂无操作"));
    return;
  }
  for (const item of items) {
    els.operationRows.append(row([
      item.id,
      labelFor(item.type),
      pill(item.status),
      formatDateTime(item.created_at),
      operationResultCell(item.result_summary),
    ]));
  }
}

async function checkReviewSafety() {
  if (!state.review || isActionBusy()) {
    return;
  }
  const jobId = state.review.job.id;
  state.reviewSafetyChecking = true;
  renderReview();
  clearNotice();
  try {
    const payload = {
      base_review_revision: state.review.versions.review_revision,
    };
    const result = await api(`/api/v1/jobs/${jobId}/review-safety-check`, {
      method: "POST",
      headers: { "Idempotency-Key": `review-safety-${randomId()}` },
      body: JSON.stringify(payload),
    });
    if (state.selectedJobId !== jobId) {
      return;
    }
    state.reviewSafetyCheck = { jobId, result };
    state.applyValidation = null;
    state.rematchPreview = null;
    state.rematchPreviewGeneration += 1;
    await loadJobDetail(jobId, {
      preserveRollback: true,
      preserveReviewSafety: true,
      preserveNotice: true,
    });
    if (state.selectedJobId === jobId) {
      showNotice(`安全检查：${labelFor(result.status)} · ${labelFor(result.token_status)}`);
    }
  } catch (error) {
    if (state.selectedJobId === jobId) {
      showNotice(error.message, true);
    }
  } finally {
    state.reviewSafetyChecking = false;
    if (state.review) {
      renderReview();
    }
  }
}

async function checkRollbackEligibility() {
  const safetyForJob =
    state.review && state.reviewSafetyCheck?.jobId === state.review.job.id
      ? state.reviewSafetyCheck.result
      : null;
  const reviewSafetyReady =
    safetyForJob &&
    safetyForJob.status === "verified" &&
    isTimedTokenCurrent(safetyForJob, "review_safety_check_token");
  if (!state.review || isActionBusy()) {
    return;
  }
  if (!reviewSafetyReady) {
    showNotice("需先通过安全检查", true);
    return;
  }
  const jobId = state.review.job.id;
  state.rollbackChecking = true;
  els.checkRollback.disabled = true;
  renderReview();
  clearNotice();
  try {
    const payload = {
      mode: "withdraw_to_review",
      scope: { kind: "all" },
      base_review_revision: state.review.versions.review_revision,
      base_move_state_revision: state.review.versions.move_state_revision,
    };
    const result = await api(`/api/v1/jobs/${jobId}/rollback-eligibility`, {
      method: "POST",
      headers: { "Idempotency-Key": `rollback-${randomId()}` },
      body: JSON.stringify(payload),
    });
    if (state.selectedJobId !== jobId) {
      return;
    }
    state.rollbackEligibility = { jobId, result };
    await loadJobDetail(jobId, {
      preserveRollback: true,
      preserveReviewSafety: true,
      preserveNotice: true,
    });
    if (state.selectedJobId === jobId) {
      showNotice(`回撤检查：${labelFor(result.status)} · ${labelFor(result.token_status)}`);
    }
  } catch (error) {
    if (state.selectedJobId === jobId) {
      showNotice(error.message, true);
    }
  } finally {
    state.rollbackChecking = false;
    if (state.review) {
      renderReview();
    }
  }
}

async function validateApply() {
  if (!state.review || isActionBusy()) {
    return;
  }
  const jobId = state.review.job.id;
  state.validateChecking = true;
  els.validateApply.disabled = true;
  renderReview();
  clearNotice();
  try {
    const payload = {
      scope: "all_for_apply",
      base_mapping_version: state.review.versions.mapping_version,
      base_move_plan_version: state.review.versions.move_plan_version,
      base_move_state_revision: state.review.versions.move_state_revision,
      base_review_revision: state.review.versions.review_revision,
    };
    const result = await api(`/api/v1/jobs/${jobId}/validate`, {
      method: "POST",
      headers: { "Idempotency-Key": `validate-${randomId()}` },
      body: JSON.stringify(payload),
    });
    if (state.selectedJobId !== jobId) {
      return;
    }
    state.applyValidation = { jobId, result };
    await loadJobDetail(jobId, {
      preserveRollback: true,
      preserveValidation: true,
      preserveNotice: true,
    });
    if (state.selectedJobId === jobId) {
      if (state.review && validationMatchesReview(result, state.review)) {
        showNotice(
          `应用验证：${labelFor(result.status)} · 阻塞 ${result.blocking_items.length}`
        );
      } else {
        showNotice("应用验证已过期，请重新验证", true);
      }
    }
  } catch (error) {
    if (state.selectedJobId === jobId) {
      showNotice(error.message, true);
    }
  } finally {
    state.validateChecking = false;
    if (state.review) {
      renderReview();
    }
  }
}

async function applyValidated() {
  const validationForJob =
    state.review &&
    state.applyValidation?.jobId === state.review.job.id &&
    validationMatchesReview(state.applyValidation.result, state.review)
      ? state.applyValidation.result
      : null;
  if (
    !state.review ||
    !validationForJob ||
    isActionBusy()
  ) {
    return;
  }
  if (validationForJob.status !== "valid") {
    showNotice("需要先通过全量验证", true);
    return;
  }
  const jobId = state.review.job.id;
  const confirmed = await confirmDangerousAction(
    applyConfirmation(validationForJob)
  );
  if (!confirmed) {
    return;
  }
  if (!ensureDetailIdle("等待刷新完成后再应用验证结果")) {
    return;
  }
  if (
    !state.review ||
    state.review.job.id !== jobId ||
    !validationMatchesReview(validationForJob, state.review)
  ) {
    showNotice("验证结果已过期，请重新验证后再应用", true);
    return;
  }
  state.applySubmitting = true;
  renderReview();
  clearNotice();
  try {
    const payload = {
      scope: "all_validated",
      validation_id: validationForJob.validation_id,
      base_mapping_version: state.review.versions.mapping_version,
      base_move_plan_version: state.review.versions.move_plan_version,
      base_move_state_revision: state.review.versions.move_state_revision,
      base_review_revision: state.review.versions.review_revision,
    };
    const result = await api(`/api/v1/jobs/${jobId}/apply`, {
      method: "POST",
      headers: { "Idempotency-Key": `apply-${randomId()}` },
      body: JSON.stringify(payload),
    });
    if (state.selectedJobId !== jobId) {
      return;
    }
    state.applyValidation = null;
    state.rematchPreview = null;
    state.rematchPreviewGeneration += 1;
    await loadJobDetail(jobId, {
      preserveRollback: true,
      preserveNotice: true,
    });
    if (state.selectedJobId === jobId) {
      showNotice(`应用完成：${result.applied_move_item_count || 0} 个移动项`);
    }
  } catch (error) {
    if (state.selectedJobId === jobId) {
      showNotice(error.message, true);
    }
  } finally {
    state.applySubmitting = false;
    if (state.review) {
      renderReview();
    }
  }
}

async function withdrawChecked() {
  const rollbackForJob =
    state.review && state.rollbackEligibility?.jobId === state.review.job.id
      ? state.rollbackEligibility.result
      : null;
  const rollbackReady =
    rollbackForJob &&
    rollbackForJob.mode === "withdraw_to_review" &&
    rollbackForJob.status === "eligible" &&
    isTimedTokenCurrent(rollbackForJob, "check_token");
  const safetyForJob =
    state.review && state.reviewSafetyCheck?.jobId === state.review.job.id
      ? state.reviewSafetyCheck.result
      : null;
  const reviewSafetyReady =
    safetyForJob &&
    safetyForJob.status === "verified" &&
    isTimedTokenCurrent(safetyForJob, "review_safety_check_token");
  if (
    !state.review ||
    !rollbackReady ||
    !reviewSafetyReady ||
    isActionBusy()
  ) {
    return;
  }
  const jobId = state.review.job.id;
  const confirmed = await confirmDangerousAction(
    withdrawConfirmation(rollbackForJob)
  );
  if (!confirmed) {
    return;
  }
  if (!ensureDetailIdle("等待刷新完成后再执行回撤")) {
    return;
  }
  if (!state.review || state.review.job.id !== jobId) {
    return;
  }
  if (
    !isTimedTokenCurrent(rollbackForJob, "check_token") ||
    !isTimedTokenCurrent(safetyForJob, "review_safety_check_token")
  ) {
    showNotice("检查已过期，请重新检查", true);
    renderReview();
    return;
  }
  state.withdrawSubmitting = true;
  renderReview();
  clearNotice();
  try {
    const payload = {
      mode: "withdraw_to_review",
      scope: { kind: "all" },
      base_review_revision: state.review.versions.review_revision,
      base_move_state_revision: state.review.versions.move_state_revision,
      check_token: rollbackForJob.check_token,
      review_safety_check_token: safetyForJob.review_safety_check_token,
    };
    const result = await api(`/api/v1/jobs/${jobId}/withdraw`, {
      method: "POST",
      headers: { "Idempotency-Key": `withdraw-${randomId()}` },
      body: JSON.stringify(payload),
    });
    if (state.selectedJobId !== jobId) {
      return;
    }
    state.rollbackEligibility = null;
    state.applyValidation = null;
    state.rematchPreview = null;
    state.rematchPreviewGeneration += 1;
    await loadJobDetail(jobId, {
      preserveNotice: true,
    });
    if (state.selectedJobId === jobId) {
      showNotice(
        `回撤完成：${result.withdrawn_move_item_count || 0} 个移动项`
      );
    }
  } catch (error) {
    if (state.selectedJobId === jobId) {
      showNotice(error.message, true);
    }
  } finally {
    state.withdrawSubmitting = false;
    if (state.review) {
      renderReview();
    }
  }
}

async function reopenFromReview() {
  const safetyForJob =
    state.review && state.reviewSafetyCheck?.jobId === state.review.job.id
      ? state.reviewSafetyCheck.result
      : null;
  const reviewSafetyReady =
    safetyForJob &&
    safetyForJob.status === "verified" &&
    isTimedTokenCurrent(safetyForJob, "review_safety_check_token");
  if (
    !state.review ||
    !reviewSafetyReady ||
    isActionBusy() ||
    !state.review.rollback_capability_hints.has_review_records
  ) {
    return;
  }
  const jobId = state.review.job.id;
  const confirmed = await confirmDangerousAction(
    reopenConfirmation(state.review)
  );
  if (!confirmed) {
    return;
  }
  if (!ensureDetailIdle("等待刷新完成后再重新打开")) {
    return;
  }
  if (!state.review || state.review.job.id !== jobId) {
    return;
  }
  if (!isTimedTokenCurrent(safetyForJob, "review_safety_check_token")) {
    showNotice("安全检查已过期，请重新检查", true);
    renderReview();
    return;
  }
  state.reopenSubmitting = true;
  renderReview();
  clearNotice();
  try {
    const payload = {
      scope: { kind: "all" },
      base_review_revision: state.review.versions.review_revision,
      base_move_state_revision: state.review.versions.move_state_revision,
      review_safety_check_token: safetyForJob.review_safety_check_token,
    };
    const result = await api(`/api/v1/jobs/${jobId}/reopen-from-review`, {
      method: "POST",
      headers: { "Idempotency-Key": `reopen-${randomId()}` },
      body: JSON.stringify(payload),
    });
    if (state.selectedJobId !== jobId) {
      return;
    }
    state.rollbackEligibility = null;
    state.applyValidation = null;
    state.rematchPreview = null;
    state.rematchPreviewGeneration += 1;
    await loadJobDetail(jobId, {
      preserveNotice: true,
    });
    if (state.selectedJobId === jobId) {
      showNotice(`重新打开完成：${result.reopened_move_item_count || 0} 个移动项`);
    }
  } catch (error) {
    if (state.selectedJobId === jobId) {
      showNotice(error.message, true);
    }
  } finally {
    state.reopenSubmitting = false;
    if (state.review) {
      renderReview();
    }
  }
}

async function previewRematch() {
  if (!state.review) {
    return;
  }
  if (isActionBusy()) {
    showNotice("等待当前操作完成后再预览重匹配", true);
    return;
  }
  const tmdbId = Number(els.tmdbInput.value);
  if (!Number.isInteger(tmdbId) || tmdbId <= 0) {
    showNotice("请输入有效的 TMDB ID", true);
    return;
  }
  const jobId = state.review.job.id;
  const previewGeneration = state.rematchPreviewGeneration + 1;
  state.rematchPreviewGeneration = previewGeneration;
  state.rematchPreviewing = true;
  els.previewRematch.disabled = true;
  clearNotice();
  try {
    const additionalInstructions = normalizedRematchInstructions();
    const payload = {
      tmdb_id: tmdbId,
      base_review_revision: state.review.versions.review_revision,
      ...(additionalInstructions
        ? { additional_instructions: additionalInstructions }
        : {}),
    };
    const result = await api(`/api/v1/jobs/${jobId}/rematch-preview`, {
      method: "POST",
      headers: { "Idempotency-Key": `rematch-${randomId()}` },
      body: JSON.stringify(payload),
    });
    if (
      state.selectedJobId !== jobId ||
      state.rematchPreviewGeneration !== previewGeneration
    ) {
      return;
    }
    state.rematchPreview = { jobId, result };
    activateTab(els.tabButtonRematch);
    await loadJobDetail(jobId, {
      preserveRollback: true,
      preserveRematch: true,
      preserveNotice: true,
    });
    if (state.selectedJobId === jobId) {
      const diff = result.diff_summary || {};
      showNotice(
        `重匹配预览：已变更 ${diff.changed || 0} · 新增 ${diff.added || 0} · 已移除 ${diff.removed || 0}`
      );
    }
  } catch (error) {
    if (state.selectedJobId === jobId) {
      showNotice(error.message, true);
    }
  } finally {
    state.rematchPreviewing = false;
    if (state.review) {
      renderReview();
    }
  }
}

async function commitRematch() {
  const previewForJob =
    state.review && state.rematchPreview?.jobId === state.review.job.id
      ? state.rematchPreview.result
      : null;
  if (!state.review || !previewForJob || isActionBusy()) {
    return;
  }
  const jobId = state.review.job.id;
  const confirmed = await confirmDangerousAction(
    rematchCommitConfirmation(previewForJob)
  );
  if (!confirmed) {
    return;
  }
  if (!ensureDetailIdle("等待刷新完成后再提交重匹配")) {
    return;
  }
  if (
    !state.review ||
    state.review.job.id !== jobId ||
    state.rematchPreview?.result !== previewForJob
  ) {
    showNotice("重匹配预览已过期，请重新预览后再提交", true);
    return;
  }
  state.rematchCommitting = true;
  els.commitRematch.disabled = true;
  clearNotice();
  try {
    const payload = {
      preview_operation_id: previewForJob.operation_id,
      tmdb_id: previewForJob.tmdb_id,
      base_review_revision: state.review.versions.review_revision,
    };
    const result = await api(`/api/v1/jobs/${jobId}/rematch-commit`, {
      method: "POST",
      headers: { "Idempotency-Key": `rematch-commit-${randomId()}` },
      body: JSON.stringify(payload),
    });
    if (state.selectedJobId !== jobId) {
      return;
    }
    state.rematchPreview = null;
    await loadJobDetail(jobId, {
      preserveRollback: true,
      preserveNotice: true,
    });
    if (state.selectedJobId === jobId) {
      showNotice(
        `重匹配已提交：${result.committed_move_item_count || 0} 个移动项`
      );
    }
  } catch (error) {
    if (state.selectedJobId === jobId) {
      showNotice(error.message, true);
    }
  } finally {
    state.rematchCommitting = false;
    if (state.review) {
      renderReview();
    }
  }
}

async function submitCorrection() {
  if (!state.review) {
    return;
  }
  if (isActionBusy()) {
    showNotice("等待当前操作完成后再提交修正", true);
    return;
  }
  const ops = buildCorrectionOps();
  if (!ops) {
    return;
  }
  const jobId = state.review.job.id;
  state.correctionSubmitting = true;
  els.submitCorrection.disabled = true;
  clearNotice();
  try {
    const payload = {
      base_review_revision: state.review.versions.review_revision,
      atomic: true,
      ops,
    };
    const result = await api(`/api/v1/jobs/${jobId}/corrections`, {
      method: "POST",
      headers: { "Idempotency-Key": `correction-${randomId()}` },
      body: JSON.stringify(payload),
    });
    if (state.selectedJobId !== jobId) {
      return;
    }
    state.rematchPreviewGeneration += 1;
    state.rematchPreview = null;
    clearCorrectionForm();
    await loadJobDetail(jobId, {
      preserveRollback: true,
      preserveNotice: true,
    });
    if (state.selectedJobId === jobId) {
      showNotice(`修正已提交：${result.op_results.length} 个操作`);
    }
  } catch (error) {
    if (state.selectedJobId === jobId) {
      showNotice(error.message, true);
    }
  } finally {
    state.correctionSubmitting = false;
    if (state.review) {
      renderReview();
    }
  }
}

function applyConfirmation(validation) {
  const preview = validation.destination_preview || [];
  return {
    title: "应用验证结果",
    summary: `将应用验证 #${validation.validation_id}，移动 ${preview.length} 个项目到计划目标。`,
    confirmLabel: "应用验证结果",
    notes: [
      `警告：${validation.warning_items?.length || 0}`,
      `校验版本：${validation.checked_review_revision || "-"}`,
    ],
    items: previewItems(preview),
  };
}

function withdrawConfirmation(result) {
  const preview = result.destination_preview || [];
  return {
    title: "执行回撤",
    summary: `将把 ${preview.length || result.affected_move_items?.length || 0} 个已移动项目转移到 review 目录。`,
    confirmLabel: "执行回撤",
    notes: [
      `模式：${labelFor(result.mode)}`,
      `回撤令牌：${labelFor(result.token_status)}`,
    ],
    items: previewItems(preview),
  };
}

function reopenConfirmation(review) {
  const items = review.move_items.filter((item) => item.status === "withdrawn");
  return {
    title: "重新打开",
    summary: `将从 review 目录重新打开 ${items.length} 个已回撤项目，状态会回到待处理。`,
    confirmLabel: "重新打开",
    notes: [`review 记录：${review.rollback_capability_hints.has_review_records ? "有" : "无"}`],
    items: confirmationItems(items, (item) => ({
      label: `移动项 ${item.id}`,
      meta: item.kind,
      fields: [
        ["review 目录", displayPath(item.current_path || item.source_path)],
        ["计划目标", displayPath(item.planned_destination_path)],
      ],
    })),
  };
}

function rematchCommitConfirmation(preview) {
  const diff = preview.diff_summary || {};
  const items = preview.diff_items || preview.move_items || [];
  return {
    title: "提交预览为计划",
    summary: `将提交 TMDB ${preview.tmdb_id} 的重匹配预览。已变更 ${diff.changed || 0}，新增 ${diff.added || 0}，已移除 ${diff.removed || 0}。`,
    confirmLabel: "提交预览",
    notes: [
      `预览操作：${preview.operation_id}`,
      `附加提示：${labelFor(instructionPreviewLabel(preview))}`,
    ],
    items: confirmationItems(items, (item) => ({
      label: `移动项 ${item.move_item_id || item.source_id || "-"}`,
      meta: item.change_type || item.status || item.kind,
      fields: [
        [
          "当前计划",
          displayPath(
            item.current_planned_destination_path ||
              item.old_destination ||
              item.current_path
          ),
        ],
        [
          "预览目标",
          displayPath(
            item.preview_destination_path ||
              item.new_planned_destination_path ||
              item.new_destination ||
              item.destination
          ),
        ],
      ],
    })),
  };
}

function previewItems(items) {
  return confirmationItems(items, (item) => ({
    label: `移动项 ${item.move_item_id || "-"}`,
    meta: item.kind || `源 ID ${item.source_id || "-"}`,
    fields: [
      ["来源", displayPath(item.from)],
      ["目标", displayPath(item.to || item.destination)],
    ],
  }));
}

function confirmationItems(items, mapper, limit = 8) {
  const mapped = items.slice(0, limit).map(mapper);
  if (items.length > limit) {
    mapped.push({
      label: `还有 ${items.length - limit} 项未显示`,
      meta: "summary",
      fields: [],
    });
  }
  return mapped;
}

function confirmDangerousAction(config) {
  if (!els.confirmDialog || typeof els.confirmDialog.showModal !== "function") {
    return Promise.resolve(
      window.confirm(`${config.title}\n\n${config.summary}`)
    );
  }
  els.confirmTitle.textContent = config.title;
  els.confirmSummary.textContent = config.summary;
  els.confirmProceed.textContent = config.confirmLabel || "确认执行";
  els.confirmDetails.replaceChildren();
  for (const note of config.notes || []) {
    const node = document.createElement("div");
    node.className = "confirm-note";
    node.textContent = note;
    els.confirmDetails.append(node);
  }
  if (config.items?.length) {
    for (const item of config.items) {
      els.confirmDetails.append(confirmDetailItem(item));
    }
  } else {
    els.confirmDetails.append(empty("没有可显示的明细"));
  }
  return new Promise((resolve) => {
    let done = false;
    const finish = (confirmed) => {
      if (done) {
        return;
      }
      done = true;
      cleanup();
      if (els.confirmDialog.open) {
        els.confirmDialog.close();
      }
      resolve(confirmed);
    };
    const onCancel = (event) => {
      event.preventDefault();
      finish(false);
    };
    const onClose = () => finish(false);
    const cleanup = () => {
      els.confirmCancel.removeEventListener("click", cancel);
      els.confirmClose.removeEventListener("click", cancel);
      els.confirmProceed.removeEventListener("click", proceed);
      els.confirmDialog.removeEventListener("cancel", onCancel);
      els.confirmDialog.removeEventListener("close", onClose);
    };
    const cancel = () => finish(false);
    const proceed = () => finish(true);
    els.confirmCancel.addEventListener("click", cancel);
    els.confirmClose.addEventListener("click", cancel);
    els.confirmProceed.addEventListener("click", proceed);
    els.confirmDialog.addEventListener("cancel", onCancel);
    els.confirmDialog.addEventListener("close", onClose);
    els.confirmDialog.showModal();
  });
}

function confirmDetailItem(item) {
  const node = document.createElement("article");
  node.className = "confirm-item";
  const header = document.createElement("header");
  const title = document.createElement("strong");
  title.textContent = item.label;
  const meta = document.createElement("span");
  meta.className = "muted";
  const rawMeta = item.meta || "";
  const metaLabel = rawMeta ? labelFor(rawMeta) : "";
  meta.textContent = metaLabel;
  if (rawMeta && metaLabel !== String(rawMeta)) {
    meta.title = String(rawMeta);
  }
  header.append(title, meta);
  node.append(header);
  if (item.fields?.length) {
    const list = document.createElement("dl");
    for (const [label, value] of item.fields) {
      const dt = document.createElement("dt");
      dt.textContent = label;
      const dd = document.createElement("dd");
      dd.textContent = value || "-";
      list.append(dt, dd);
    }
    node.append(list);
  }
  return node;
}

function displayPath(value) {
  if (value === null || value === undefined) {
    return "-";
  }
  if (typeof value === "string") {
    return value;
  }
  return value.display_path || value.rel_path || value.path || compactJson(value);
}

function buildCorrectionOps() {
  const ops = [];
  const targetId = optionalPositiveInt(els.correctionTargetId.value);
  const season = optionalNonNegativeInt(els.correctionSeason.value);
  const start = optionalPositiveInt(els.correctionStart.value);
  const end = optionalPositiveInt(els.correctionEnd.value);
  const episodeValues = [targetId, season, start, end];
  const hasEpisodeEdit = episodeValues.some((value) => value !== null);
  if (hasEpisodeEdit && episodeValues.some((value) => value === null)) {
    showNotice("修正 episode 需要填写 Target ID、Season、Start、End", true);
    return null;
  }
  if (hasEpisodeEdit) {
    ops.push({
      op: "update_episode",
      episode_target_id: targetId,
      season,
      episode_start: start,
      episode_end: end,
    });
  }

  const subtitleLinkId = optionalPositiveInt(els.correctionSubtitleLinkId.value);
  if (subtitleLinkId !== null) {
    ops.push({
      op: "override_subtitle_variant",
      subtitle_link_id: subtitleLinkId,
      variant: els.correctionVariant.value,
    });
  }
  if (!ops.length) {
    showNotice("请至少填写一个修正项", true);
    return null;
  }
  return ops;
}

function renderRollbackDetails(result) {
  els.rollbackDetails.replaceChildren();
  updateDisclosureSummary(
    els.rollbackDetailsDisclosure,
    els.rollbackDetailsSummary,
    "回撤明细",
    rollbackDisclosureText(result),
    rollbackProblemCount(result) > 0,
  );
  if (!result) {
    return;
  }
  const summary = document.createElement("div");
  summary.className = "detail-item";
  summary.textContent = `${labelFor(result.mode)} · ${labelFor(result.status)} · ${labelFor(result.token_status)}`;
  els.rollbackDetails.append(summary);
  for (const item of result.destination_preview || []) {
    const node = document.createElement("div");
    node.className = "detail-item";
    node.textContent = `#${item.move_item_id}: ${item.from?.display_path || "-"} -> ${item.to?.display_path || "-"}`;
    els.rollbackDetails.append(node);
  }
  appendRollbackProblems("missing", result.missing_files);
  appendRollbackProblems("incomplete", result.incomplete_records);
  appendRollbackProblems("identity", result.identity_mismatches);
  appendRollbackProblems("blocking", result.target_conflicts);
}

function renderRematchPreview(result) {
  els.rematchRows.replaceChildren();
  if (!result) {
    renderSelectedTmdbBlock(els.rematchTmdbDetails, null);
    renderDefinitionList(els.rematchInfo, {
      status: "not checked",
      tmdb: "-",
      instructions: "unused",
      changed: 0,
      added: 0,
      removed: 0,
    });
    els.rematchRows.append(emptyTableRow(5, "先生成重匹配预览"));
    return;
  }
  const diff = result.diff_summary || {};
  renderSelectedTmdbBlock(els.rematchTmdbDetails, result.selected_tmdb_info);
  renderDefinitionList(els.rematchInfo, {
    status: result.status,
    operation: result.operation_id,
    baseRev: result.base_review_revision,
    tmdb: result.tmdb_id,
    title: result.selected_tmdb_info?.title_zh_cn || "-",
    originalName: result.selected_tmdb_info?.original_name || "-",
    year: result.selected_tmdb_info?.year || "-",
    instructions: instructionPreviewLabel(result),
    validation: result.validation_summary?.status || "-",
    changed: diff.changed || 0,
    added: diff.added || 0,
    removed: diff.removed || 0,
    unchanged: diff.unchanged || 0,
  });
  const items = result.diff_items || result.move_items || [];
  if (!items.length) {
    els.rematchRows.append(emptyTableRow(5, "无变更"));
    return;
  }
  for (const item of items) {
    els.rematchRows.append(row([
      item.source_id,
      labelFor(item.kind),
      pill(item.change_type, rematchChangeMode(item.change_type)),
      item.current_planned_destination_path
        ? pathCell(item.current_planned_destination_path)
        : "-",
      item.preview_destination_path ? pathCell(item.preview_destination_path) : "-",
    ]));
  }
}

function rematchChangeMode(changeType) {
  if (changeType === "removed") {
    return "error";
  }
  if (changeType === "changed" || changeType === "added") {
    return "warn";
  }
  return "";
}

function appendRollbackProblems(label, items) {
  for (const item of items || []) {
    const node = document.createElement("div");
    node.className = "detail-item";
    node.textContent = `${labelFor(label)}：${compactJson(item)}`;
    els.rollbackDetails.append(node);
  }
}

function renderDefinitionList(target, data) {
  target.replaceChildren();
  for (const [key, value] of Object.entries(data)) {
    appendDefinitionItem(target, key, value);
  }
}

function appendDefinitionItem(target, key, value) {
  const dt = document.createElement("dt");
  dt.textContent = labelFor(key);
  const dd = document.createElement("dd");
  if (value instanceof Node) {
    dd.append(value);
  } else {
    dd.textContent = labelFor(value);
  }
  target.append(dt, dd);
}

function row(cells) {
  const tr = document.createElement("tr");
  for (const value of cells) {
    const td = document.createElement("td");
    if (value instanceof Node) {
      td.append(value);
    } else {
      td.textContent = value ?? "-";
    }
    tr.append(td);
  }
  return tr;
}

function emptyTableRow(colspan, text) {
  const tr = document.createElement("tr");
  const td = document.createElement("td");
  td.colSpan = colspan;
  td.append(empty(text));
  tr.append(td);
  return tr;
}

function operationResultCell(value) {
  const details = document.createElement("details");
  details.className = "operation-result";
  const summary = document.createElement("summary");
  summary.textContent = operationResultSummary(value);
  const pre = document.createElement("pre");
  pre.textContent = JSON.stringify(value || {}, null, 2);
  details.append(summary, pre);
  return details;
}

function operationResultSummary(value) {
  if (!value || typeof value !== "object") {
    return "查看 JSON";
  }
  const parts = [];
  for (const [key, label] of [
    ["status", "状态"],
    ["checked", "已检查"],
    ["applied_move_item_count", "已应用"],
    ["withdrawn_move_item_count", "已回撤"],
    ["reopened_move_item_count", "已重新打开"],
    ["committed_move_item_count", "已提交"],
    ["changed", "已变更"],
    ["added", "新增"],
    ["removed", "已移除"],
  ]) {
    if (value[key] !== undefined && value[key] !== null) {
      parts.push(`${label} ${labelFor(value[key])}`);
    }
  }
  if (value.item_result_summary && typeof value.item_result_summary === "object") {
    const summary = value.item_result_summary;
    for (const [key, label] of [
      ["checked", "已检查"],
      ["succeeded", "成功"],
      ["failed", "失败"],
    ]) {
      if (summary[key] !== undefined && summary[key] !== null) {
        parts.push(`${label} ${summary[key]}`);
      }
    }
  }
  return parts.slice(0, 3).join(" / ") || "查看 JSON";
}

function formatDateTime(value) {
  if (!value) {
    return "-";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return date.toLocaleString();
}

function pill(text, mode = "") {
  const span = document.createElement("span");
  span.className = `pill ${mode}`.trim();
  const label = labelFor(text);
  span.textContent = label;
  if (text && label !== String(text)) {
    span.title = String(text);
  }
  return span;
}

function pathCell(ref) {
  const span = document.createElement("span");
  span.className = "path";
  const text = ref?.display_path || ref?.rel_path || "-";
  span.textContent = text;
  if (text !== "-") {
    span.title = text;
  }
  return span;
}

function empty(text) {
  const node = document.createElement("div");
  node.className = "empty-state muted";
  node.textContent = text;
  return node;
}

function formatEpisode(item) {
  const base = `S${pad2(item.season)}E${pad2(item.episode_start)}`;
  if (item.episode_end === item.episode_start) {
    return base;
  }
  return `${base}-E${pad2(item.episode_end)}`;
}

function pad2(value) {
  return String(value).padStart(2, "0");
}

function validationMatchesReview(validation, review) {
  return (
    validation.validated_mapping_version === review.versions.mapping_version &&
    validation.validated_move_plan_version === review.versions.move_plan_version &&
    validation.checked_move_state_revision === review.versions.move_state_revision
  );
}

function instructionPreviewLabel(result) {
  if (!result?.additional_instructions_present) {
    return "unused";
  }
  const digest = result.additional_instructions_digest
    ? result.additional_instructions_digest.slice(0, 12)
    : "-";
  return `used · ${result.additional_instructions_length || 0} 字 · ${digest}`;
}

function normalizedRematchInstructions() {
  const value = els.rematchInstructions.value.trim();
  return value || "";
}

function optionalPositiveInt(value) {
  const normalized = value.trim();
  if (!normalized) {
    return null;
  }
  const parsed = Number(normalized);
  if (!Number.isInteger(parsed) || parsed <= 0) {
    return null;
  }
  return parsed;
}

function optionalNonNegativeInt(value) {
  const normalized = value.trim();
  if (!normalized) {
    return null;
  }
  const parsed = Number(normalized);
  if (!Number.isInteger(parsed) || parsed < 0) {
    return null;
  }
  return parsed;
}

function clearCorrectionForm() {
  els.correctionTargetId.value = "";
  els.correctionSeason.value = "";
  els.correctionStart.value = "";
  els.correctionEnd.value = "";
  els.correctionSubtitleLinkId.value = "";
  els.correctionVariant.value = "detected";
}

function compactJson(value) {
  const text = JSON.stringify(value || {});
  if (text.length <= 80) {
    return text;
  }
  return `${text.slice(0, 77)}...`;
}

function randomId() {
  if (globalThis.crypto?.randomUUID) {
    return globalThis.crypto.randomUUID();
  }
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function showNotice(message, error = false) {
  els.notice.classList.toggle("error", error);
  els.notice.setAttribute("role", error ? "alert" : "status");
  els.notice.setAttribute("aria-live", error ? "assertive" : "polite");
  els.notice.classList.remove("hidden");
  els.notice.textContent = message;
}

function clearNotice() {
  els.notice.textContent = "";
  els.notice.classList.add("hidden");
  els.notice.classList.remove("error");
  els.notice.setAttribute("role", "status");
  els.notice.setAttribute("aria-live", "polite");
}

function clearDetail(options = {}) {
  state.detailLoadingJobId = null;
  if (state.tokenExpiryTimer) {
    clearTimeout(state.tokenExpiryTimer);
    state.tokenExpiryTimer = null;
  }
  state.review = null;
  state.safety = null;
  state.operations = [];
  els.jobTitle.textContent = "选择一个 job";
  els.jobMeta.textContent = "";
  els.refreshReview.disabled = state.selectedJobId === null;
  setButtonState(els.checkReviewSafety, true);
  setButtonState(els.checkRollback, true);
  setButtonState(els.withdrawChecked, true);
  setButtonState(els.reopenFromReview, true);
  els.validateApply.disabled = true;
  els.applyValidated.disabled = true;
  els.previewRematch.disabled = true;
  els.commitRematch.disabled = true;
  els.submitCorrection.disabled = true;
  if (!options.preserveTmdbInput) {
    els.tmdbInput.value = "";
    els.rematchInstructions.value = "";
    state.tmdbInputSeededJobId = null;
    state.tmdbInputEditedJobId = null;
  }
  clearCorrectionForm();
  for (const target of [
    els.readiness,
    els.readinessDetails,
    els.tmdbInfo,
    els.tmdbDetails,
    els.tmdbCandidates,
    els.safetyInfo,
    els.validationInfo,
    els.validationDetails,
    els.rollbackInfo,
    els.episodeRows,
    els.subtitleRows,
    els.fileRows,
    els.moveRows,
    els.operationRows,
    els.rematchInfo,
    els.rematchTmdbDetails,
    els.rematchRows,
  ]) {
    target.replaceChildren();
  }
  els.evidenceList.replaceChildren();
  els.rollbackDetails.replaceChildren();
  renderNoJobEmptyState();
  updateDisclosureSummary(
    els.readinessDetailsDisclosure,
    els.readinessDetailsSummary,
    "就绪明细",
    "选择 job 后显示",
    false,
  );
  updateDisclosureSummary(
    els.tmdbCandidatesDisclosure,
    els.tmdbCandidatesSummary,
    "TMDB 候选",
    "选择 job 后显示",
    false,
  );
  updateDisclosureSummary(
    els.validationDetailsDisclosure,
    els.validationDetailsSummary,
    "验证明细",
    "选择 job 后显示",
    false,
  );
  updateDisclosureSummary(
    els.rollbackDetailsDisclosure,
    els.rollbackDetailsSummary,
    "回撤明细",
    "选择 job 后显示",
    false,
  );
  setRollbackStep(
    els.rollbackSafetyStep,
    els.rollbackSafetyStatus,
    "未检查",
    "pending",
  );
  setRollbackStep(
    els.rollbackEligibilityStep,
    els.rollbackEligibilityStatus,
    "未检查",
    "pending",
  );
  setRollbackStep(
    els.rollbackExecuteStep,
    els.rollbackExecuteStatus,
    "等待前置条件",
    "pending",
  );
}

function renderNoJobEmptyState() {
  const text = "选择 job 后显示";
  els.readinessDetails.append(empty(text));
  els.tmdbDetails.append(empty(text));
  els.tmdbCandidates.append(empty(text));
  els.validationDetails.append(empty(text));
  els.rollbackDetails.append(empty(text));
  els.episodeRows.append(emptyTableRow(5, text));
  els.subtitleRows.append(emptyTableRow(5, text));
  els.fileRows.append(emptyTableRow(6, text));
  els.moveRows.append(emptyTableRow(7, text));
  els.operationRows.append(emptyTableRow(5, text));
  els.rematchTmdbDetails.append(empty(text));
  els.rematchRows.append(emptyTableRow(5, text));
  els.evidenceList.append(empty(text));
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

clearDetail({ preserveTmdbInput: true });
startJobsAutoRefresh();
loadRuntimeAndJobs();
