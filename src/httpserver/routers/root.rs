use crate::httpserver::handlers::{current_config, device_is_cuda, embedding, root};

use axum::error_handling::HandleErrorLayer;
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{BoxError, Router};

use std::time::Duration;
use tower::ServiceBuilder;
use tower_http::{compression::CompressionLayer, trace::TraceLayer};

pub fn router_root() -> Router {
    let tracer = TraceLayer::new_for_http();
    let middleware_stack = ServiceBuilder::new()
        .layer(tracer)
        .layer(CompressionLayer::new())
        .layer(HandleErrorLayer::new(handle_timeout_error))
        .layer(tower::timeout::TimeoutLayer::new(Duration::from_secs(2)))
        .into_inner();

    let root = Router::new()
        .route("/health", get(root))
        .route("/health", post(root));

    let task_router = Router::new()
        // .route(
        //     "/template/transfer/oss2oss",
        //     get(task_template_transfer_oss2oss),
        // )
        .layer(middleware_stack.clone());

    let api = Router::new()
        .route("/v1/currentconfig", post(current_config))
        .route("/v1/deviceiscuda", post(device_is_cuda))
        .route("/v1/embedding", post(embedding))
        .layer(middleware_stack.clone())
        .nest("/v1/task", task_router);

    return root.nest("/api", api);
}

async fn handle_timeout_error(err: BoxError) -> (StatusCode, String) {
    if err.is::<tower::timeout::error::Elapsed>() {
        (StatusCode::REQUEST_TIMEOUT, "Request timeout".to_string())
    } else {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Unhandled internal error: {}", err),
        )
    }
}
