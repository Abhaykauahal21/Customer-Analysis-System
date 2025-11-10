import React, { useState, useEffect } from 'react';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell } from 'recharts';
import { getClusters } from '../services/api';
import LoadingSkeleton from '../components/LoadingSkeleton';

const COLORS = ['#0ea5e9', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899'];

const Segmentation = () => {
  const [clusterData, setClusterData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchClusters();
  }, []);

  const fetchClusters = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await getClusters();
      
      if (!data) {
        throw new Error('No cluster data received from server');
      }
      
      setClusterData(data);
    } catch (err) {
      console.error('Error fetching clusters:', err);
      const errorMessage = err.error || err.message || 'Failed to load cluster data. Please ensure the backend is running and ML pipeline has been executed.';
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div>
        <h1 className="text-3xl font-bold text-gray-800 dark:text-white mb-6">
          Customer Segmentation
        </h1>
        <LoadingSkeleton className="h-96" />
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

  // Group data by cluster
  const clusters = {};
  if (clusterData?.pca_data) {
    clusterData.pca_data.forEach((point) => {
      const cluster = point.cluster;
      if (!clusters[cluster]) {
        clusters[cluster] = [];
      }
      clusters[cluster].push(point);
    });
  }

  const clusterSeries = Object.keys(clusters).map((cluster) => ({
    name: `Cluster ${cluster}`,
    data: clusters[cluster],
  }));

  return (
    <div>
      <h1 className="text-3xl font-bold text-gray-800 dark:text-white mb-6">
        Customer Segmentation (PCA + K-Means)
      </h1>

      {/* Info Cards */}
      {clusterData?.explained_variance && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
            <h3 className="text-lg font-semibold text-gray-800 dark:text-white mb-2">
              PCA Explained Variance
            </h3>
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-gray-600 dark:text-gray-400">PC1:</span>
                <span className="font-semibold text-gray-800 dark:text-white">
                  {(clusterData.explained_variance[0] * 100).toFixed(2)}%
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600 dark:text-gray-400">PC2:</span>
                <span className="font-semibold text-gray-800 dark:text-white">
                  {(clusterData.explained_variance[1] * 100).toFixed(2)}%
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600 dark:text-gray-400">Total:</span>
                <span className="font-semibold text-gray-800 dark:text-white">
                  {(clusterData.explained_variance.reduce((a, b) => a + b, 0) * 100).toFixed(2)}%
                </span>
              </div>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
            <h3 className="text-lg font-semibold text-gray-800 dark:text-white mb-2">
              Cluster Statistics
            </h3>
            <div className="space-y-2">
              {Object.keys(clusters).map((cluster) => (
                <div key={cluster} className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <div
                      className="w-4 h-4 rounded-full"
                      style={{ backgroundColor: COLORS[cluster % COLORS.length] }}
                    ></div>
                    <span className="text-gray-600 dark:text-gray-400">Cluster {cluster}:</span>
                  </div>
                  <span className="font-semibold text-gray-800 dark:text-white">
                    {clusters[cluster].length} customers
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Scatter Plot */}
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
        <h2 className="text-xl font-semibold text-gray-800 dark:text-white mb-4">
          PCA Visualization with K-Means Clusters
        </h2>
        <ResponsiveContainer width="100%" height={600}>
          <ScatterChart>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis
              type="number"
              dataKey="x"
              name="PC1"
              label={{ value: 'Principal Component 1', position: 'insideBottom', offset: -5 }}
            />
            <YAxis
              type="number"
              dataKey="y"
              name="PC2"
              label={{ value: 'Principal Component 2', angle: -90, position: 'insideLeft' }}
            />
            <Tooltip cursor={{ strokeDasharray: '3 3' }} />
            <Legend />
            {clusterSeries.map((series, index) => (
              <Scatter
                key={series.name}
                name={series.name}
                data={series.data}
                fill={COLORS[index % COLORS.length]}
              />
            ))}
            {/* Cluster Centers */}
            {clusterData?.cluster_centers && (
              <Scatter
                name="Cluster Centers"
                data={clusterData.cluster_centers}
                fill="#000"
                shape="star"
              />
            )}
          </ScatterChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default Segmentation;

