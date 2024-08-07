use std::collections::HashMap;

use axum::Json;
use qdrant_client::qdrant::{ScoredPoint, SearchPoints, Value};
use uuid::Uuid;

use crate::{
    embedding::{answer::answer, embedding_setence, retriever::retriever},
    httpserver::{
        exception::{AppError, AppErrorType},
        module::{module_retriever::RespRetriever, ReqContent, ReqRetriever, Response},
    },
};

use super::HandlerResult;

pub async fn handler_embedding(Json(req): Json<ReqContent>) -> HandlerResult<Vec<Vec<f32>>> {
    match embedding_setence(&req.content).await {
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

pub async fn handler_retriever(Json(req): Json<ReqRetriever>) -> HandlerResult<Vec<RespRetriever>> {
    match retriever(&req.content, req.limit).await {
        Ok(r) => {
            let mut vec_resp = vec![];
            for p in r.result {
                let id = match p.id {
                    Some(pid) => match pid.point_id_options {
                        Some(pido) => match pido {
                            qdrant_client::qdrant::point_id::PointIdOptions::Num(n) => {
                                n.to_string()
                            }
                            qdrant_client::qdrant::point_id::PointIdOptions::Uuid(s) => s,
                        },
                        None => "".to_string(),
                    },
                    None => "".to_string(),
                };

                let payload = p.payload.clone();
                let score = p.score;
                let resp = RespRetriever { id, payload, score };
                vec_resp.push(resp);
            }
            Ok(Json(Response::ok(vec_resp)))
        }
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

pub async fn handler_answer(Json(req): Json<ReqRetriever>) -> HandlerResult<String> {
    match answer(&req.content, req.limit as usize) {
        Ok(s) => Ok(Json(Response::ok(s))),
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
