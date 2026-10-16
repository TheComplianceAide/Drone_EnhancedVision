const path = require('node:path');
const root = '/Users/randyblasik/Documents/Drone_EnhancedVision';
const config = require(path.join(root, 'node_media_server_config.js'));
config.rtmp.port = 21935;
config.http.port = 18080;
process.env.DRONE_NMS_STREAM_PATH = '/live/qa_tonight';
process.env.DRONE_NMS_STATE_PATH = path.join(process.env.QA_RUN_DIR || '/tmp/drone-tonight-gpu-20260904', 'nms_state.json');
require(path.join(root, 'nms_local_server.js'));
