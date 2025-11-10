import React, { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { getModels } from '../services/api';
import LoadingSkeleton, { CardSkeleton } from '../components/LoadingSkeleton';

const Dashboard = () => {
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [summary, setSummary] = useState(null);

  useEffect(() => {
    fetchModels();
  }, []);

  const fetchModels = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await getModels();
      
      if (!data || !data.models) {
        throw new Error('Invalid response format from server');
      }
      
      setModels(data.models || []);
      
      // Calculate summary if not provided
      if (data.summary) {
        setSummary(data.summary);
      } else {
        // Calculate from models array
        const totalAcc = data.models.reduce((sum, m) => sum + (m.accuracy || 0), 0);
        const avgAcc = totalAcc / data.models.length;
        const maxAcc = Math.max(...data.models.map(m => m.accuracy || 0));
        const minAcc = Math.min(...data.models.map(m => m.accuracy || 0));
        setSummary({
          total_accuracy: totalAcc,
          average_accuracy: avgAcc,
          max_accuracy: maxAcc,
          min_accuracy: minAcc,
          num_models: data.models.length
        });
      }
    } catch (err) {
      console.error('Error fetching models:', err);
      const errorMessage = err.error || err.message || 'Failed to load model data. Please ensure the backend is running.';
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div>
        <h1 className="text-3xl font-bold text-gray-800 dark:text-white mb-6">
          Model Performance Dashboard
        </h1>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          {[1, 2, 3, 4].map((i) => (
            <CardSkeleton key={i} />
          ))}
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4">
        <p className="text-red-800 dark:text-red-200">{error}</p>
      </div>
    );
  }

  const bestModel = models[0]; // Already sorted by F1 score

  return (
    <div>
      <h1 className="text-3xl font-bold text-gray-800 dark:text-white mb-6">
        Model Performance Dashboard
      </h1>

      {/* Summary Statistics */}
      {summary && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <div className="bg-gradient-to-br from-blue-500 to-blue-600 text-white rounded-xl shadow-lg p-6">
            <h3 className="text-sm font-semibold opacity-90 mb-2">Total Accuracy (Sum)</h3>
            <p className="text-3xl font-bold">{(summary.total_accuracy * 100).toFixed(2)}%</p>
            <p className="text-xs opacity-75 mt-2">{summary.num_models} models combined</p>
          </div>
          <div className="bg-gradient-to-br from-green-500 to-green-600 text-white rounded-xl shadow-lg p-6">
            <h3 className="text-sm font-semibold opacity-90 mb-2">Average Accuracy</h3>
            <p className="text-3xl font-bold">{(summary.average_accuracy * 100).toFixed(2)}%</p>
            <p className="text-xs opacity-75 mt-2">Across all models</p>
          </div>
          <div className="bg-gradient-to-br from-purple-500 to-purple-600 text-white rounded-xl shadow-lg p-6">
            <h3 className="text-sm font-semibold opacity-90 mb-2">Best Model Accuracy</h3>
            <p className="text-3xl font-bold">{(summary.max_accuracy * 100).toFixed(2)}%</p>
            <p className="text-xs opacity-75 mt-2">Highest performing</p>
          </div>
          <div className="bg-gradient-to-br from-orange-500 to-orange-600 text-white rounded-xl shadow-lg p-6">
            <h3 className="text-sm font-semibold opacity-90 mb-2">Number of Models</h3>
            <p className="text-3xl font-bold">{summary.num_models}</p>
            <p className="text-xs opacity-75 mt-2">Trained models</p>
          </div>
        </div>
      )}

      {/* Best Model Card */}
      {bestModel && (
        <div className="bg-gradient-to-r from-primary-500 to-primary-600 text-white rounded-xl shadow-lg p-6 mb-8">
          <h2 className="text-xl font-semibold mb-2">🏆 Best Performing Model</h2>
          <p className="text-2xl font-bold">{bestModel.name}</p>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-4">
            <div>
              <p className="text-sm opacity-90">Accuracy</p>
              <p className="text-xl font-bold">{(bestModel.accuracy * 100).toFixed(2)}%</p>
            </div>
            <div>
              <p className="text-sm opacity-90">Precision</p>
              <p className="text-xl font-bold">{(bestModel.precision * 100).toFixed(2)}%</p>
            </div>
            <div>
              <p className="text-sm opacity-90">Recall</p>
              <p className="text-xl font-bold">{(bestModel.recall * 100).toFixed(2)}%</p>
            </div>
            <div>
              <p className="text-sm opacity-90">F1-Score</p>
              <p className="text-xl font-bold">{(bestModel.f1_score * 100).toFixed(2)}%</p>
            </div>
          </div>
        </div>
      )}

      {/* Model Comparison Chart */}
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 mb-8">
        <h2 className="text-xl font-semibold text-gray-800 dark:text-white mb-4">
          Model Comparison
        </h2>
        <ResponsiveContainer width="100%" height={400}>
          <BarChart data={models}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Bar dataKey="accuracy" fill="#0ea5e9" name="Accuracy" />
            <Bar dataKey="precision" fill="#10b981" name="Precision" />
            <Bar dataKey="recall" fill="#f59e0b" name="Recall" />
            <Bar dataKey="f1_score" fill="#8b5cf6" name="F1-Score" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Model Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {models.map((model, index) => (
          <div
            key={index}
            className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 hover:shadow-xl transition-shadow"
          >
            <h3 className="text-lg font-semibold text-gray-800 dark:text-white mb-4">
              {model.name}
            </h3>
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-gray-600 dark:text-gray-400">Accuracy:</span>
                <span className="font-semibold text-gray-800 dark:text-white">
                  {(model.accuracy * 100).toFixed(2)}%
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600 dark:text-gray-400">Precision:</span>
                <span className="font-semibold text-gray-800 dark:text-white">
                  {(model.precision * 100).toFixed(2)}%
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600 dark:text-gray-400">Recall:</span>
                <span className="font-semibold text-gray-800 dark:text-white">
                  {(model.recall * 100).toFixed(2)}%
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600 dark:text-gray-400">F1-Score:</span>
                <span className="font-semibold text-gray-800 dark:text-white">
                  {(model.f1_score * 100).toFixed(2)}%
                </span>
              </div>
              {model.roc_auc && (
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">ROC-AUC:</span>
                  <span className="font-semibold text-gray-800 dark:text-white">
                    {model.roc_auc.toFixed(4)}
                  </span>
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default Dashboard;

