use axum::Json;

use crate::{
    embedding::{get_token, model_device_is_cuda},
    httpserver::{
        exception::{AppError, AppErrorType},
        module::{ReqContent, Response},
    },
};

use super::HandlerResult;

pub async fn device_is_cuda() -> HandlerResult<bool> {
    Ok(Json(Response::ok(model_device_is_cuda().await)))

    // match service_task_create(&mut task) {
    //     Ok(id) => Ok(Json(Response::ok(TaskId {
    //         task_id: id.to_string(),
    //     }))),
    //     Err(e) => {
    //         let err = AppError {
    //             message: Some(e.to_string()),
    //             cause: None,
    //             error_type: AppErrorType::UnknowErr,
    //         };
    //         return Err(err);
    //     }
    // }
}

pub async fn embedding(Json(req): Json<ReqContent>) -> HandlerResult<Vec<Vec<f32>>> {
    match get_token(&req.content).await {
        Ok(token) => Ok(Json(Response::ok(token))),
        Err(e) => {
            let err = AppError {
                message: Some(e.to_string()),
                cause: None,
                error_type: AppErrorType::UnknowErr,
            };
            return Err(err);
        }
    }
}
