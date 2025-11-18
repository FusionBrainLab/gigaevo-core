import axios from 'axios';

// Create axios instance with base configuration
const api = axios.create({
  baseURL: 'http://localhost:8081/api',
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for logging
api.interceptors.request.use(
  (config) => {
    console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    console.error('API Request Error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => {
    console.log(`API Response: ${response.status} ${response.config.url}`);
    return response;
  },
  (error) => {
    console.error('API Response Error:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

/**
 * Get all available stages from the registry
 */
export async function getStages() {
  try {
    const response = await api.get('/stages');
    return Array.isArray(response.data) ? response.data : Object.values(response.data);
  } catch (error) {
    throw new Error(`Failed to fetch stages: ${error.response?.data?.detail || error.message}`);
  }
}

/**
 * Get a specific stage by name
 */
export async function getStage(stageName) {
  try {
    const response = await api.get(`/stages/${stageName}`);
    return response.data;
  } catch (error) {
    throw new Error(`Failed to fetch stage ${stageName}: ${error.response?.data?.detail || error.message}`);
  }
}

/**
 * Export a DAG as PipelineBuilder code
 */
export async function exportDAG(dagData) {
  try {
    const response = await api.post('/export-dag', dagData);
    return response.data;
  } catch (error) {
    throw new Error(`Failed to export DAG: ${error.response?.data?.detail || error.message}`);
  }
}

/**
 * Validate a DAG structure
 */
export async function validateDAG(dagData) {
  try {
    const response = await api.post('/validate-dag', dagData);
    return response.data;
  } catch (error) {
    throw new Error(`Failed to validate DAG: ${error.response?.data?.detail || error.message}`);
  }
}

/**
 * Export a DAG as YAML configuration
 */
export async function exportYAML(dagData) {
  try {
    const response = await api.post('/export-yaml', dagData);
    return response.data;
  } catch (error) {
    throw new Error(`Failed to export YAML: ${error.response?.data?.detail || error.message}`);
  }
}

/**
 * List available Hydra YAML pipeline configs
 */
export async function listYAMLConfigs() {
  try {
    const response = await api.get('/yaml-configs');
    return response.data;
  } catch (error) {
    throw new Error(`Failed to list YAML configs: ${error.response?.data?.detail || error.message}`);
  }
}

/**
 * Load and parse a Hydra YAML config
 */
export async function loadYAMLConfig(yamlPath) {
  try {
    const response = await api.post('/load-yaml', { yaml_path: yamlPath });
    return response.data;
  } catch (error) {
    throw new Error(`Failed to load YAML config: ${error.response?.data?.detail || error.message}`);
  }
}

export default api;
