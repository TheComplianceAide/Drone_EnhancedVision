#!/usr/bin/env node
"use strict";

// Local Node Media Server runner with a zero-decode publisher heartbeat.
// The launcher reads the heartbeat instead of repeatedly opening an FFmpeg
// subscriber merely to decide whether the aircraft is publishing.

const crypto = require("node:crypto");
const fs = require("node:fs");
const path = require("node:path");

const NodeMediaServer = require("node-media-server");
const config = require("./node_media_server_config.js");

const statePath = process.env.DRONE_NMS_STATE_PATH ||
  path.join(__dirname, "logs", "nms_state.json");
const streamPath = process.env.DRONE_NMS_STREAM_PATH || "/live/mavic3";
const runId = crypto.randomUUID();

fs.mkdirSync(path.dirname(statePath), { recursive: true });

let publisher = null;
let lastBytes = 0;
let lastByteAdvanceAtMs = 0;
let stopping = false;

function finiteNumber(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n : 0;
}

function statePayload(serverAlive = true) {
  const now = Date.now();
  let pub = null;
  if (publisher && publisher.isPublisher && publisher.streamPath === streamPath) {
    const inBytes = finiteNumber(publisher.inBytes);
    if (inBytes !== lastBytes) {
      lastBytes = inBytes;
      lastByteAdvanceAtMs = now;
    }
    pub = {
      session_id: String(publisher.id || ""),
      ip: String(publisher.ip || ""),
      connected_at_ms: finiteNumber(publisher.createTime),
      in_bytes: inBytes,
      last_byte_advance_at_ms: lastByteAdvanceAtMs,
      video_width: finiteNumber(publisher.videoWidth),
      video_height: finiteNumber(publisher.videoHeight),
      video_framerate: finiteNumber(publisher.videoFramerate),
      video_codec: finiteNumber(publisher.videoCodec)
    };
  }
  return {
    schema: 1,
    run_id: runId,
    pid: process.pid,
    server_alive: Boolean(serverAlive),
    stream_path: streamPath,
    updated_at_ms: now,
    publisher: pub
  };
}

function writeState(serverAlive = true) {
  const tmp = `${statePath}.${process.pid}.tmp`;
  try {
    fs.writeFileSync(tmp, `${JSON.stringify(statePayload(serverAlive))}\n`, "utf8");
    fs.renameSync(tmp, statePath);
  } catch (err) {
    try { fs.unlinkSync(tmp); } catch (_) { /* best effort */ }
    console.error(`[nms-heartbeat] ${err && err.message ? err.message : err}`);
  }
}

const nms = new NodeMediaServer(config);

nms.on("postPublish", (session) => {
  // NMS 4.2.8 emits postPublish before duplicate-publisher validation and
  // before isPublisher is set. Defer one turn and accept only the session
  // that survived validation.
  setImmediate(() => {
    if (!session.isPublisher || session.streamPath !== streamPath) return;
    publisher = session;
    lastBytes = finiteNumber(session.inBytes);
    lastByteAdvanceAtMs = Date.now();
    writeState(true);
  });
});

nms.on("donePublish", (session) => {
  if (publisher && session.id === publisher.id) {
    publisher = null;
    lastBytes = 0;
    lastByteAdvanceAtMs = 0;
    writeState(true);
  }
});

function shutdown(signal) {
  if (stopping) return;
  stopping = true;
  publisher = null;
  writeState(false);
  console.log(`[nms-heartbeat] stopping on ${signal}`);
  process.exit(0);
}

process.on("SIGINT", () => shutdown("SIGINT"));
process.on("SIGTERM", () => shutdown("SIGTERM"));

nms.run();
writeState(true);
setInterval(() => writeState(true), 250);
