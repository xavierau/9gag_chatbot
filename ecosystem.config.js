const fs = require('fs');
const path = require('path');

// Load .env file
const envPath = path.join(__dirname, '.env');
const env = {};
if (fs.existsSync(envPath)) {
  fs.readFileSync(envPath, 'utf8').split('\n').forEach(line => {
    const match = line.match(/^([^#=]+)=(.*)$/);
    if (match) {
      env[match[1].trim()] = match[2].trim();
    }
  });
}

const appName = env.APP_NAME || 'chatbot';
const port = env.PORT || '8000';

module.exports = {
  apps: [
    {
      name: appName,
      script: '.venv/bin/python',
      args: `-m uvicorn app.main:app --host 0.0.0.0 --port ${port} --workers 1`,
      interpreter: 'none',
      cwd: __dirname,
      env_file: '.env',
      instances: 1,
      autorestart: true,
      watch: false,
      max_memory_restart: '1G',
      log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
      error_file: `logs/${appName}-error.log`,
      out_file: `logs/${appName}-out.log`,
      merge_logs: true
    }
  ]
};