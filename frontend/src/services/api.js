import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 30000, // 30 second timeout
});

// Request interceptor for debugging
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
    return response;
  },
  (error) => {
    console.error('API Response Error:', error);
    
    // Handle network errors
    if (!error.response) {
      return Promise.reject({
        error: 'Network error: Could not connect to the server. Please ensure the backend is running.',
        message: error.message,
        type: 'network_error'
      });
    }
    
    // Handle HTTP errors
    const errorData = error.response.data || {};
    return Promise.reject({
      error: errorData.error || error.message || 'An unexpected error occurred',
      message: errorData.message || error.message,
      type: errorData.type || 'api_error',
      status: error.response.status
    });
  }
);

export const predictPurchase = async (customerData) => {
  try {
    const response = await api.post('/api/predict', customerData);
    return response.data;
  } catch (error) {
    // Error is already formatted by interceptor
    throw error;
  }
};

export const getClusters = async () => {
  try {
    const response = await api.get('/api/clusters');
    return response.data;
  } catch (error) {
    throw error;
  }
};

export const getModels = async () => {
  try {
    const response = await api.get('/api/models');
    return response.data;
  } catch (error) {
    throw error;
  }
};

export const getFeatures = async () => {
  try {
    const response = await api.get('/api/features');
    return response.data;
  } catch (error) {
    throw error;
  }
};

export default api;

