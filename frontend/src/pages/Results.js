import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';

const Results = () => {
  const navigate = useNavigate();
  const [result, setResult] = useState(null);

  useEffect(() => {
    try {
      const storedResult = localStorage.getItem('predictionResult');
      if (storedResult) {
        const parsedResult = JSON.parse(storedResult);
        if (parsedResult && typeof parsedResult === 'object') {
          setResult(parsedResult);
        } else {
          console.error('Invalid result format');
          navigate('/prediction');
        }
      } else {
        // Redirect to prediction if no result
        navigate('/prediction');
      }
    } catch (err) {
      console.error('Error parsing result:', err);
      navigate('/prediction');
    }
  }, [navigate]);

  if (!result) {
    return (
      <div className="flex items-center justify-center h-64">
        <p className="text-gray-600 dark:text-gray-400">Loading results...</p>
      </div>
    );
  }

  const predictionText = result.prediction === 1 ? 'Likely to Purchase' : 'Unlikely to Purchase';
  const predictionColor = result.prediction === 1 ? 'green' : 'red';
  
  // Get confidence level
  const confidence = result.confidence || (result.probability?.purchase || result.probability?.no_purchase || 0.5);
  const confidencePercent = (confidence * 100).toFixed(1);

  return (
    <div>
      <h1 className="text-3xl font-bold text-gray-800 dark:text-white mb-6">
        Prediction Results
      </h1>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Main Prediction Card */}
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
          <h2 className="text-xl font-semibold text-gray-800 dark:text-white mb-4">
            Purchase Prediction
          </h2>
          <div className="text-center">
            <div
              className={`inline-block px-6 py-3 rounded-lg text-2xl font-bold mb-4 ${
                predictionColor === 'green'
                  ? 'bg-green-100 dark:bg-green-900/30 text-green-800 dark:text-green-200'
                  : 'bg-red-100 dark:bg-red-900/30 text-red-800 dark:text-red-200'
              }`}
            >
              {predictionText}
            </div>
            {result.probability && (
              <div className="mt-4">
                <p className="text-gray-600 dark:text-gray-400 mb-2">
                  Confidence: {confidencePercent}%
                  {result.confidence_threshold && (
                    <span className="text-xs text-gray-500 dark:text-gray-400 block">
                      (Threshold: {result.confidence_threshold * 100}%)
                    </span>
                  )}
                </p>
                <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-4">
                  <div
                    className={`h-4 rounded-full ${
                      predictionColor === 'green' ? 'bg-green-500' : 'bg-red-500'
                    }`}
                    style={{
                      width: `${Math.min(100, Math.max(0, confidence * 100))}%`,
                    }}
                  ></div>
                </div>
                <p className="text-xs text-gray-500 dark:text-gray-400 mt-2">
                  {result.prediction === 1 
                    ? `Purchase probability: ${((result.probability.purchase || 0) * 100).toFixed(1)}%`
                    : `No purchase probability: ${((result.probability.no_purchase || 0) * 100).toFixed(1)}%`
                  }
                </p>
              </div>
            )}
          </div>
        </div>

        {/* Probability Breakdown */}
        {result.probability && (
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
            <h2 className="text-xl font-semibold text-gray-800 dark:text-white mb-4">
              Probability Breakdown
            </h2>
            <div className="space-y-4">
              <div>
                <div className="flex justify-between mb-2">
                  <span className="text-gray-600 dark:text-gray-400">Purchase:</span>
                  <span className="font-semibold text-gray-800 dark:text-white">
                    {((result.probability.purchase || 0) * 100).toFixed(2)}%
                  </span>
                </div>
                <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-3">
                  <div
                    className="bg-green-500 h-3 rounded-full"
                    style={{ width: `${Math.min(100, Math.max(0, (result.probability.purchase || 0) * 100))}%` }}
                  ></div>
                </div>
              </div>
              <div>
                <div className="flex justify-between mb-2">
                  <span className="text-gray-600 dark:text-gray-400">No Purchase:</span>
                  <span className="font-semibold text-gray-800 dark:text-white">
                    {((result.probability.no_purchase || 0) * 100).toFixed(2)}%
                  </span>
                </div>
                <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-3">
                  <div
                    className="bg-red-500 h-3 rounded-full"
                    style={{ width: `${Math.min(100, Math.max(0, (result.probability.no_purchase || 0) * 100))}%` }}
                  ></div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Cluster Assignment */}
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
          <h2 className="text-xl font-semibold text-gray-800 dark:text-white mb-4">
            Customer Segment
          </h2>
          <div className="text-center">
            <div className="text-4xl font-bold text-primary-500 mb-2">
              Cluster {result.cluster}
            </div>
            <p className="text-gray-600 dark:text-gray-400 mb-4">
              This customer belongs to cluster {result.cluster}
            </p>
            {result.cluster_center_distance !== undefined && result.cluster_center_distance !== null && (
              <p className="text-sm text-gray-500 dark:text-gray-500">
                Distance from cluster center: {Number(result.cluster_center_distance).toFixed(4)}
              </p>
            )}
          </div>
        </div>

        {/* PCA Coordinates */}
        {result.pca_coordinates && (
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
            <h2 className="text-xl font-semibold text-gray-800 dark:text-white mb-4">
              PCA Coordinates
            </h2>
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-gray-600 dark:text-gray-400">Principal Component 1:</span>
                <span className="font-semibold text-gray-800 dark:text-white">
                  {Number(result.pca_coordinates.x || 0).toFixed(4)}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600 dark:text-gray-400">Principal Component 2:</span>
                <span className="font-semibold text-gray-800 dark:text-white">
                  {Number(result.pca_coordinates.y || 0).toFixed(4)}
                </span>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* All Models Predictions */}
      {result.all_predictions && Object.keys(result.all_predictions).length > 0 && (
        <div className="mt-8">
          <h2 className="text-2xl font-bold text-gray-800 dark:text-white mb-6">
            Predictions from All Models
          </h2>
          
          {/* Consensus Card */}
          {result.consensus && (
            <div className="bg-gradient-to-r from-indigo-500 to-purple-600 text-white rounded-xl shadow-lg p-6 mb-6">
              <h3 className="text-xl font-semibold mb-2">🎯 Model Consensus</h3>
              <p className="text-lg mb-2">
                {result.consensus.agreement || `${result.consensus.vote_count}/${result.consensus.total_models} models agree`}
              </p>
              <div className={`inline-block px-4 py-2 rounded-lg text-lg font-bold ${
                result.consensus.prediction === 1
                  ? 'bg-green-100 text-green-800'
                  : 'bg-red-100 text-red-800'
              }`}>
                {result.consensus.prediction === 1 ? 'Likely to Purchase' : 'Unlikely to Purchase'}
              </div>
            </div>
          )}
          
          {/* All Models Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {Object.entries(result.all_predictions).map(([modelName, modelResult]) => {
              const isPurchase = modelResult.prediction === 1;
              const purchaseProb = (modelResult.probability?.purchase || 0) * 100;
              const noPurchaseProb = (modelResult.probability?.no_purchase || 0) * 100;
              const confidence = (modelResult.confidence || 0) * 100;
              
              return (
                <div
                  key={modelName}
                  className={`bg-white dark:bg-gray-800 rounded-xl shadow-lg p-5 border-2 ${
                    isPurchase
                      ? 'border-green-500 dark:border-green-400'
                      : 'border-red-500 dark:border-red-400'
                  }`}
                >
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="text-lg font-semibold text-gray-800 dark:text-white">
                      {modelName}
                    </h3>
                    <span
                      className={`px-3 py-1 rounded-full text-sm font-bold ${
                        isPurchase
                          ? 'bg-green-100 dark:bg-green-900/30 text-green-800 dark:text-green-200'
                          : 'bg-red-100 dark:bg-red-900/30 text-red-800 dark:text-red-200'
                      }`}
                    >
                      {isPurchase ? 'Purchase' : 'No Purchase'}
                    </span>
                  </div>
                  
                  <div className="space-y-3">
                    <div>
                      <div className="flex justify-between text-sm mb-1">
                        <span className="text-gray-600 dark:text-gray-400">Purchase:</span>
                        <span className="font-semibold text-gray-800 dark:text-white">
                          {purchaseProb.toFixed(1)}%
                        </span>
                      </div>
                      <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                        <div
                          className="bg-green-500 h-2 rounded-full"
                          style={{ width: `${Math.min(100, Math.max(0, purchaseProb))}%` }}
                        ></div>
                      </div>
                    </div>
                    
                    <div>
                      <div className="flex justify-between text-sm mb-1">
                        <span className="text-gray-600 dark:text-gray-400">No Purchase:</span>
                        <span className="font-semibold text-gray-800 dark:text-white">
                          {noPurchaseProb.toFixed(1)}%
                        </span>
                      </div>
                      <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                        <div
                          className="bg-red-500 h-2 rounded-full"
                          style={{ width: `${Math.min(100, Math.max(0, noPurchaseProb))}%` }}
                        ></div>
                      </div>
                    </div>
                    
                    <div className="pt-2 border-t border-gray-200 dark:border-gray-700">
                      <div className="flex justify-between text-xs">
                        <span className="text-gray-500 dark:text-gray-400">Confidence:</span>
                        <span className="font-semibold text-gray-800 dark:text-white">
                          {confidence.toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Action Buttons */}
      <div className="mt-6 flex justify-center space-x-4">
        <button
          onClick={() => navigate('/prediction')}
          className="px-6 py-3 bg-primary-500 text-white rounded-lg font-semibold hover:bg-primary-600 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 transition-colors"
        >
          Make Another Prediction
        </button>
        <button
          onClick={() => navigate('/segmentation')}
          className="px-6 py-3 bg-gray-500 text-white rounded-lg font-semibold hover:bg-gray-600 focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-offset-2 transition-colors"
        >
          View Segmentation
        </button>
      </div>
    </div>
  );
};

export default Results;

