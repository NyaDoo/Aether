use axum::body::Body;
use axum::http::Response;

use crate::gateway::ai_pipeline::planner::plan_builders::{
    LocalStreamPlanAndReport, LocalSyncPlanAndReport,
};
use crate::gateway::scheduler::record_local_request_candidate_status;
use crate::gateway::{
    execute_execution_runtime_stream, execute_execution_runtime_sync, AppState,
    GatewayControlDecision, GatewayError,
};

pub(crate) trait LocalPlanAndReport {
    fn plan(&self) -> &aether_contracts::ExecutionPlan;

    fn report_kind(&self) -> Option<String>;

    fn report_context(&self) -> Option<serde_json::Value>;
}

impl LocalPlanAndReport for LocalSyncPlanAndReport {
    fn plan(&self) -> &aether_contracts::ExecutionPlan {
        &self.plan
    }

    fn report_kind(&self) -> Option<String> {
        self.report_kind.clone()
    }

    fn report_context(&self) -> Option<serde_json::Value> {
        self.report_context.clone()
    }
}

impl LocalPlanAndReport for LocalStreamPlanAndReport {
    fn plan(&self) -> &aether_contracts::ExecutionPlan {
        &self.plan
    }

    fn report_kind(&self) -> Option<String> {
        self.report_kind.clone()
    }

    fn report_context(&self) -> Option<serde_json::Value> {
        self.report_context.clone()
    }
}

pub(crate) async fn execute_sync_plan_and_reports<T>(
    state: &AppState,
    parts: &http::request::Parts,
    trace_id: &str,
    decision: &GatewayControlDecision,
    plan_kind: &str,
    plan_and_reports: Vec<T>,
) -> Result<Option<Response<Body>>, GatewayError>
where
    T: LocalPlanAndReport,
{
    let mut remaining = plan_and_reports.into_iter();
    while let Some(plan_and_report) = remaining.next() {
        if let Some(response) = execute_execution_runtime_sync(
            state,
            parts.uri.path(),
            plan_and_report.plan().clone(),
            trace_id,
            decision,
            plan_kind,
            plan_and_report.report_kind(),
            plan_and_report.report_context(),
        )
        .await?
        {
            mark_unused_local_candidates(state, remaining.collect()).await;
            return Ok(Some(response));
        }
    }

    Ok(None)
}

pub(crate) async fn execute_stream_plan_and_reports<T>(
    state: &AppState,
    trace_id: &str,
    decision: &GatewayControlDecision,
    plan_kind: &str,
    plan_and_reports: Vec<T>,
) -> Result<Option<Response<Body>>, GatewayError>
where
    T: LocalPlanAndReport,
{
    let mut remaining = plan_and_reports.into_iter();
    while let Some(plan_and_report) = remaining.next() {
        if let Some(response) = execute_execution_runtime_stream(
            state,
            plan_and_report.plan().clone(),
            trace_id,
            decision,
            plan_kind,
            plan_and_report.report_kind(),
            plan_and_report.report_context(),
        )
        .await?
        {
            mark_unused_local_candidates(state, remaining.collect()).await;
            return Ok(Some(response));
        }
    }

    Ok(None)
}

pub(crate) async fn mark_unused_local_candidates<T>(state: &AppState, remaining: Vec<T>)
where
    T: LocalPlanAndReport,
{
    for plan_and_report in remaining {
        record_local_request_candidate_status(
            state,
            plan_and_report.plan(),
            plan_and_report.report_context().as_ref(),
            aether_data::repository::candidates::RequestCandidateStatus::Unused,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        .await;
    }
}
