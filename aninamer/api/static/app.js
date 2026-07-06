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

const els = {
  runtimeStatus: document.querySelector("#runtimeStatus"),
  authForm: document.querySelector("#authForm"),
  apiToken: document.querySelector("#apiToken"),
  rematchForm: document.querySelector("#rematchForm"),
  tmdbInput: document.querySelector("#tmdbInput"),
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
  tmdbInfo: document.querySelector("#tmdbInfo"),
  safetyInfo: document.querySelector("#safetyInfo"),
  validationInfo: document.querySelector("#validationInfo"),
  rollbackInfo: document.querySelector("#rollbackInfo"),
  episodeRows: document.querySelector("#episodeRows"),
  subtitleRows: document.querySelector("#subtitleRows"),
  fileRows: document.querySelector("#fileRows"),
  evidenceList: document.querySelector("#evidenceList"),
  operationRows: document.querySelector("#operationRows"),
  moveRows: document.querySelector("#moveRows"),
  rollbackDetails: document.querySelector("#rollbackDetails"),
  rematchInfo: document.querySelector("#rematchInfo"),
  rematchRows: document.querySelector("#rematchRows"),
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
  state.rematchPreview = null;
  state.rematchPreviewGeneration += 1;
  if (state.review) {
    renderReview();
  }
});

document.querySelectorAll(".tab").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((item) => item.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach((item) => item.classList.remove("active"));
    button.classList.add("active");
    document.querySelector(`#tab-${button.dataset.tab}`).classList.add("active");
  });
});

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

async function loadRuntimeAndJobs() {
  clearNotice();
  try {
    const [runtime, jobs] = await Promise.all([
      api("/api/v1/runtime"),
      api("/api/v1/jobs"),
    ]);
    els.runtimeStatus.textContent = `${runtime.watch_root_keys.length} roots`;
    state.jobs = jobs.items;
    renderJobs();
    if (state.selectedJobId && state.jobs.some((job) => job.id === state.selectedJobId)) {
      await loadJobDetail(state.selectedJobId);
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
    showNotice(error.message, true);
  }
}

function renderJobs() {
  els.jobList.replaceChildren();
  for (const job of state.jobs) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `job-item${job.id === state.selectedJobId ? " active" : ""}`;
    button.innerHTML = `
      <strong>${escapeHtml(job.series_name)}</strong>
      <span>${job.id} · ${job.status} · ${job.video_moves_count}/${job.subtitle_moves_count}</span>
    `;
    button.addEventListener("click", () => loadJobDetail(job.id));
    els.jobList.append(button);
  }
}

async function loadJobDetail(jobId, options = {}) {
  const requestId = state.detailRequestId + 1;
  state.detailRequestId = requestId;
  if (!options.preserveNotice) {
    clearNotice();
  }
  state.selectedJobId = jobId;
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
  clearDetail();
  renderJobs();
  els.refreshReview.disabled = false;
  try {
    const [review, safety, operations] = await Promise.all([
      api(`/api/v1/jobs/${jobId}/review`),
      api(`/api/v1/jobs/${jobId}/safety`),
      api(`/api/v1/jobs/${jobId}/operations`),
    ]);
    if (state.detailRequestId !== requestId || state.selectedJobId !== jobId) {
      return;
    }
    state.review = review;
    state.safety = safety;
    state.operations = operations.items;
    renderReview();
  } catch (error) {
    if (state.detailRequestId !== requestId || state.selectedJobId !== jobId) {
      return;
    }
    clearDetail();
    showNotice(error.message, true);
  }
}

function renderReview() {
  const { review, safety } = state;
  if (!review) {
    return;
  }
  els.jobTitle.textContent = review.job.series_name;
  els.jobMeta.textContent = `job ${review.job.id} · ${review.job.status} · rev ${review.review_revision}`;
  const writeBusy =
    state.applySubmitting ||
    state.withdrawSubmitting ||
    state.reopenSubmitting ||
    state.rematchCommitting ||
    state.correctionSubmitting;
  const actionBusy = writeBusy || state.rematchPreviewing || state.reviewSafetyChecking;
  const safetyForJob =
    state.reviewSafetyCheck?.jobId === review.job.id
      ? state.reviewSafetyCheck.result
      : null;
  const reviewSafetyReady =
    safetyForJob &&
    safetyForJob.status === "verified" &&
    safetyForJob.token_status === "current" &&
    safetyForJob.review_safety_check_token;
  els.checkReviewSafety.disabled = actionBusy;
  els.checkRollback.disabled =
    state.rollbackChecking ||
    actionBusy ||
    !review.rollback_capability_hints.has_moved_items;
  els.validateApply.disabled =
    state.validateChecking ||
    actionBusy ||
    !review.move_items.some((item) => item.status === "proposed");
  els.previewRematch.disabled =
    actionBusy;
  const previewForJob =
    state.rematchPreview?.jobId === review.job.id ? state.rematchPreview.result : null;
  const preferredTmdbId =
    previewForJob?.tmdb_id || review.selected_tmdb_info.selected_tmdb_id;
  if (!els.tmdbInput.value && preferredTmdbId) {
    els.tmdbInput.value = preferredTmdbId;
  }
  renderDefinitionList(els.readiness, {
    status: review.readiness_summary.status || "unknown",
    blocking: review.readiness_summary.blocking_count,
    warning: review.readiness_summary.warning_count,
    unmapped: review.readiness_summary.unmapped_video_count,
    orphanSubs: review.readiness_summary.unassociated_subtitle_count,
  });
  renderDefinitionList(els.tmdbInfo, {
    tmdb: review.selected_tmdb_info.selected_tmdb_id || "-",
    title: review.selected_tmdb_info.title_zh_cn || "-",
    candidates: review.selected_tmdb_info.candidates.length,
    seasons: review.selected_tmdb_info.seasons.length,
    specials: review.selected_tmdb_info.specials.length,
  });
  renderDefinitionList(els.safetyInfo, {
    watcher: safety?.watcher_status?.status || "-",
    proof: safety?.review_exclusion_proof?.status || review.review_exclusion_proof.status,
    token: reviewSafetyReady ? "current" : "-",
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
  els.applyValidated.disabled =
    actionBusy ||
    state.validateChecking ||
    !validationForJob ||
    validationForJob.status !== "valid";
  const rollbackForJob =
    state.rollbackEligibility?.jobId === review.job.id
      ? state.rollbackEligibility.result
      : null;
  const rollbackReady =
    rollbackForJob &&
    rollbackForJob.mode === "withdraw_to_review" &&
    rollbackForJob.status === "eligible" &&
    rollbackForJob.token_status === "current" &&
    rollbackForJob.check_token;
  els.withdrawChecked.disabled =
    actionBusy ||
    state.rollbackChecking ||
    !rollbackReady ||
    !reviewSafetyReady;
  els.reopenFromReview.disabled =
    actionBusy ||
    state.rollbackChecking ||
    !review.rollback_capability_hints.has_review_records ||
    !reviewSafetyReady;
  renderDefinitionList(els.rollbackInfo, {
    hint: review.rollback_capability_hints.has_moved_items ? "has moved items" : "no moved items",
    last: review.last_rollback_check_summary.status,
    mode: review.last_rollback_check_summary.mode || "-",
    token: rollbackReady ? "current" : "-",
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
  els.commitRematch.disabled =
    state.validateChecking ||
    writeBusy ||
    state.rematchPreviewing ||
    !previewForJob;
  els.submitCorrection.disabled =
    state.validateChecking ||
    writeBusy ||
    state.rematchPreviewing ||
    !review.move_items.every((item) => item.status === "proposed");
}

function renderEpisodeRows(items) {
  els.episodeRows.replaceChildren();
  for (const item of items) {
    els.episodeRows.append(row([
      item.id,
      item.video_source_id,
      formatEpisode(item),
      pill(item.status),
      item.subtitle_link_ids.join(", ") || "-",
    ]));
  }
}

function renderSubtitleRows(items) {
  els.subtitleRows.replaceChildren();
  for (const item of items) {
    els.subtitleRows.append(row([
      item.id,
      item.episode_target_id,
      item.subtitle_source_id,
      item.variant_override || item.variant_detected,
      pill(item.status),
    ]));
  }
}

function renderFileRows(items) {
  els.fileRows.replaceChildren();
  for (const item of items) {
    els.fileRows.append(row([
      item.id,
      item.kind,
      pathCell(item.original_path),
      pathCell(item.current_path),
      pill(item.final_disposition, item.final_disposition === "failed" ? "error" : ""),
      item.ignored_reason || item.detected_subtitle_variant || "-",
    ]));
  }
}

function renderMoveRows(items) {
  els.moveRows.replaceChildren();
  for (const item of items) {
    els.moveRows.append(row([
      item.id,
      item.source_id,
      item.kind,
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
    node.innerHTML = `
      <header>
        <strong>${escapeHtml(item.evidence_kind || "evidence")}</strong>
        <span class="pill">${escapeHtml(item.confidence || "unknown")}</span>
      </header>
      <div>${escapeHtml(item.summary || "")}</div>
    `;
    els.evidenceList.append(node);
  }
}

function renderOperations(items) {
  els.operationRows.replaceChildren();
  for (const item of items) {
    els.operationRows.append(row([
      item.id,
      item.type,
      pill(item.status),
      item.created_at,
      compactJson(item.result_summary),
    ]));
  }
}

async function checkReviewSafety() {
  if (
    !state.review ||
    state.reviewSafetyChecking ||
    state.rollbackChecking ||
    state.withdrawSubmitting ||
    state.reopenSubmitting ||
    state.applySubmitting ||
    state.validateChecking ||
    state.rematchPreviewing ||
    state.rematchCommitting ||
    state.correctionSubmitting
  ) {
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
      showNotice(`review safety: ${result.status} · ${result.token_status}`);
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
  if (
    !state.review ||
    state.reviewSafetyChecking ||
    state.rollbackChecking ||
    state.withdrawSubmitting ||
    state.reopenSubmitting ||
    state.reviewSafetyChecking ||
    state.rematchPreviewing ||
    state.rematchCommitting ||
    state.correctionSubmitting
  ) {
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
      showNotice(`rollback check: ${result.status} · ${result.token_status}`);
    }
  } catch (error) {
    showNotice(error.message, true);
  } finally {
    state.rollbackChecking = false;
    if (state.review) {
      renderReview();
    }
  }
}

async function validateApply() {
  if (
    !state.review ||
    state.validateChecking ||
    state.applySubmitting ||
    state.withdrawSubmitting ||
    state.reopenSubmitting ||
    state.reviewSafetyChecking ||
    state.rematchPreviewing ||
    state.rematchCommitting ||
    state.correctionSubmitting
  ) {
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
          `apply validation: ${result.status} · blocking ${result.blocking_items.length}`
        );
      } else {
        showNotice("apply validation is stale after review changed", true);
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
    state.applySubmitting ||
    state.withdrawSubmitting ||
    state.reopenSubmitting ||
    state.rematchPreviewing ||
    state.rematchCommitting ||
    state.correctionSubmitting
  ) {
    return;
  }
  if (validationForJob.status !== "valid") {
    showNotice("需要先通过全量验证", true);
    return;
  }
  const jobId = state.review.job.id;
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
      showNotice(`apply succeeded: ${result.applied_move_item_count || 0} moves`);
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
    rollbackForJob.token_status === "current" &&
    rollbackForJob.check_token;
  const safetyForJob =
    state.review && state.reviewSafetyCheck?.jobId === state.review.job.id
      ? state.reviewSafetyCheck.result
      : null;
  const reviewSafetyReady =
    safetyForJob &&
    safetyForJob.status === "verified" &&
    safetyForJob.token_status === "current" &&
    safetyForJob.review_safety_check_token;
  if (
    !state.review ||
    !rollbackReady ||
    !reviewSafetyReady ||
    state.withdrawSubmitting ||
    state.reopenSubmitting ||
    state.reviewSafetyChecking ||
    state.rematchPreviewing ||
    state.rematchCommitting ||
    state.correctionSubmitting ||
    state.rollbackChecking
  ) {
    return;
  }
  if (state.applySubmitting || state.validateChecking) {
    showNotice("等待当前 apply 操作完成后再执行回撤", true);
    return;
  }
  const jobId = state.review.job.id;
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
        `withdraw succeeded: ${result.withdrawn_move_item_count || 0} moves`
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
    safetyForJob.token_status === "current" &&
    safetyForJob.review_safety_check_token;
  if (
    !state.review ||
    !reviewSafetyReady ||
    state.reopenSubmitting ||
    state.withdrawSubmitting ||
    state.reviewSafetyChecking ||
    state.applySubmitting ||
    state.validateChecking ||
    state.rollbackChecking ||
    state.rematchPreviewing ||
    state.rematchCommitting ||
    state.correctionSubmitting ||
    !state.review.rollback_capability_hints.has_review_records
  ) {
    return;
  }
  const jobId = state.review.job.id;
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
      showNotice(`reopened: ${result.reopened_move_item_count || 0} moves`);
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
  if (!state.review || state.rematchPreviewing) {
    return;
  }
  if (
    state.applySubmitting ||
    state.withdrawSubmitting ||
    state.reopenSubmitting ||
    state.reviewSafetyChecking ||
    state.rematchCommitting ||
    state.correctionSubmitting
  ) {
    showNotice("等待当前 apply 操作完成后再预览重匹配", true);
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
    const payload = {
      tmdb_id: tmdbId,
      base_review_revision: state.review.versions.review_revision,
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
    await loadJobDetail(jobId, {
      preserveRollback: true,
      preserveRematch: true,
      preserveNotice: true,
    });
    if (state.selectedJobId === jobId) {
      const diff = result.diff_summary || {};
      showNotice(
        `rematch preview: changed ${diff.changed || 0} · added ${diff.added || 0} · removed ${diff.removed || 0}`
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
  if (!state.review || !previewForJob || state.rematchCommitting) {
    return;
  }
  if (
    state.validateChecking ||
    state.applySubmitting ||
    state.withdrawSubmitting ||
    state.reopenSubmitting ||
    state.reviewSafetyChecking ||
    state.rematchPreviewing ||
    state.correctionSubmitting
  ) {
    showNotice("等待当前 apply 操作完成后再提交重匹配", true);
    return;
  }
  const jobId = state.review.job.id;
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
        `rematch committed: ${result.committed_move_item_count || 0} move items`
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
  if (!state.review || state.correctionSubmitting) {
    return;
  }
  if (
    state.validateChecking ||
    state.applySubmitting ||
    state.withdrawSubmitting ||
    state.reopenSubmitting ||
    state.reviewSafetyChecking ||
    state.rematchPreviewing ||
    state.rematchCommitting
  ) {
    showNotice("等待当前 apply 操作完成后再提交修正", true);
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
      showNotice(`correction committed: ${result.op_results.length} ops`);
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
  if (!result) {
    return;
  }
  const summary = document.createElement("div");
  summary.className = "detail-item";
  summary.textContent = `${result.mode} · ${result.status} · ${result.token_status}`;
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
}

function renderRematchPreview(result) {
  els.rematchRows.replaceChildren();
  if (!result) {
    renderDefinitionList(els.rematchInfo, {
      status: "not checked",
      tmdb: "-",
      changed: 0,
      added: 0,
      removed: 0,
    });
    return;
  }
  const diff = result.diff_summary || {};
  renderDefinitionList(els.rematchInfo, {
    status: result.status,
    operation: result.operation_id,
    baseRev: result.base_review_revision,
    tmdb: result.tmdb_id,
    title: result.selected_tmdb_info?.title_zh_cn || "-",
    validation: result.validation_summary?.status || "-",
    changed: diff.changed || 0,
    added: diff.added || 0,
    removed: diff.removed || 0,
    unchanged: diff.unchanged || 0,
  });
  const items = result.diff_items || result.move_items || [];
  if (!items.length) {
    els.rematchRows.append(row(["-", "-", "no changes", "-", "-"]));
    return;
  }
  for (const item of items) {
    els.rematchRows.append(row([
      item.source_id,
      item.kind,
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
    node.textContent = `${label}: ${compactJson(item)}`;
    els.rollbackDetails.append(node);
  }
}

function renderDefinitionList(target, data) {
  target.replaceChildren();
  for (const [key, value] of Object.entries(data)) {
    const dt = document.createElement("dt");
    dt.textContent = key;
    const dd = document.createElement("dd");
    dd.textContent = value ?? "-";
    target.append(dt, dd);
  }
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

function pill(text, mode = "") {
  const span = document.createElement("span");
  span.className = `pill ${mode}`.trim();
  span.textContent = text || "-";
  return span;
}

function pathCell(ref) {
  const span = document.createElement("span");
  span.className = "path";
  span.textContent = ref?.display_path || ref?.rel_path || "-";
  return span;
}

function empty(text) {
  const node = document.createElement("div");
  node.className = "muted";
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
  els.notice.textContent = message;
  els.notice.classList.toggle("error", error);
  els.notice.classList.remove("hidden");
}

function clearNotice() {
  els.notice.textContent = "";
  els.notice.classList.add("hidden");
  els.notice.classList.remove("error");
}

function clearDetail() {
  state.review = null;
  state.safety = null;
  state.operations = [];
  els.jobTitle.textContent = "选择一个 job";
  els.jobMeta.textContent = "";
  els.refreshReview.disabled = state.selectedJobId === null;
  els.checkReviewSafety.disabled = true;
  els.checkRollback.disabled = true;
  els.withdrawChecked.disabled = true;
  els.reopenFromReview.disabled = true;
  els.validateApply.disabled = true;
  els.applyValidated.disabled = true;
  els.previewRematch.disabled = true;
  els.commitRematch.disabled = true;
  els.submitCorrection.disabled = true;
  els.tmdbInput.value = "";
  clearCorrectionForm();
  for (const target of [
    els.readiness,
    els.tmdbInfo,
    els.safetyInfo,
    els.validationInfo,
    els.rollbackInfo,
    els.episodeRows,
    els.subtitleRows,
    els.fileRows,
    els.moveRows,
    els.operationRows,
    els.rematchInfo,
    els.rematchRows,
  ]) {
    target.replaceChildren();
  }
  els.evidenceList.replaceChildren();
  els.rollbackDetails.replaceChildren();
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

loadRuntimeAndJobs();
