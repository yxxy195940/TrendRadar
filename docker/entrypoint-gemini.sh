#!/bin/bash
set -e

# 保存环境变量
env >> /etc/environment

GEMINI_CRON_SCHEDULE_VALUE=${GEMINI_CRON_SCHEDULE:-"0 9 * * *"}
GEMINI_TASK_CMD_VALUE=${GEMINI_TASK_CMD:-"/usr/local/bin/python gemini_runner.py"}

case "${GEMINI_RUN_MODE:-cron}" in
"once")
    echo "🔄 Gemini 单次执行"
    exec ${GEMINI_TASK_CMD_VALUE}
    ;;
"cron")
    echo "${GEMINI_CRON_SCHEDULE_VALUE} cd /app && ${GEMINI_TASK_CMD_VALUE}" > /tmp/gemini-crontab

    echo "📅 生成的 Gemini crontab 内容:"
    cat /tmp/gemini-crontab

    if ! /usr/local/bin/supercronic -test /tmp/gemini-crontab; then
        echo "❌ Gemini crontab 格式验证失败"
        exit 1
    fi

    if [ "${GEMINI_IMMEDIATE_RUN:-false}" = "true" ]; then
        echo "▶️ 立即执行 Gemini 任务"
        ${GEMINI_TASK_CMD_VALUE}
    fi

    echo "⏰ 启动 Gemini supercronic: ${GEMINI_CRON_SCHEDULE_VALUE}"
    echo "🎯 Gemini supercronic 将作为 PID 1 运行"

    exec /usr/local/bin/supercronic -passthrough-logs /tmp/gemini-crontab
    ;;
*)
    exec "$@"
    ;;
esac
