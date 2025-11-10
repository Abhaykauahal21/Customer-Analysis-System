import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { predictPurchase, getFeatures } from '../services/api';

const Prediction = () => {
  const navigate = useNavigate();
  const [features, setFeatures] = useState([]);
  const [formData, setFormData] = useState({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchFeatures();
  }, []);

  const fetchFeatures = async () => {
    try {
      setError(null);
      const data = await getFeatures();
      
      if (!data || !data.features) {
        throw new Error('Invalid response format from server');
      }
      
      setFeatures(data.features || []);
      
      // Initialize form data with default values
      const defaults = {};
      data.features.forEach((feature) => {
        defaults[feature.name] = feature.min || 0;
      });
      setFormData(defaults);
    } catch (err) {
      console.error('Error fetching features:', err);
      const errorMessage = err.error || err.message || 'Failed to load features. Please ensure the backend is running.';
      setError(errorMessage);
    }
  };

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({
      ...formData,
      [name]: parseFloat(value) || 0,
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      // Validate that all required fields are filled
      const requiredFields = features.filter(f => f.name);
      const missingFields = requiredFields.filter(f => 
        formData[f.name] === undefined || formData[f.name] === null || formData[f.name] === ''
      );
      
      if (missingFields.length > 0) {
        throw new Error(`Please fill in all required fields: ${missingFields.map(f => f.name).join(', ')}`);
      }
      
      const result = await predictPurchase(formData);
      
      if (!result) {
        throw new Error('No prediction result received from server');
      }
      
      // Store result in localStorage to pass to Results page
      localStorage.setItem('predictionResult', JSON.stringify(result));
      navigate('/results');
    } catch (err) {
      console.error('Error making prediction:', err);
      const errorMessage = err.error || err.message || 'Failed to make prediction. Please check your input and try again.';
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  // Group features into categories
  const basicFeatures = features.filter((f) =>
    ['Age', 'Income', 'Kidhome', 'Teenhome', 'Recency'].includes(f.name)
  );
  const spendingFeatures = features.filter((f) => f.name.startsWith('Mnt'));
  const purchaseFeatures = features.filter((f) => f.name.startsWith('Num'));
  const otherFeatures = features.filter(
    (f) =>
      !basicFeatures.includes(f) &&
      !spendingFeatures.includes(f) &&
      !purchaseFeatures.includes(f)
  );

  return (
    <div>
      <h1 className="text-3xl font-bold text-gray-800 dark:text-white mb-6">
        Purchase Likelihood Prediction
      </h1>

      {error && (
        <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4 mb-6">
          <p className="text-red-800 dark:text-red-200">{error}</p>
        </div>
      )}

      <form onSubmit={handleSubmit} className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
        {/* Basic Information */}
        <div className="mb-8">
          <h2 className="text-xl font-semibold text-gray-800 dark:text-white mb-4">
            Basic Information
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {basicFeatures.map((feature) => (
              <div key={feature.name}>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  {feature.name}
                </label>
                <input
                  type="number"
                  name={feature.name}
                  value={formData[feature.name] || ''}
                  onChange={handleChange}
                  min={feature.min}
                  max={feature.max}
                  className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent bg-white dark:bg-gray-700 text-gray-800 dark:text-white"
                  required
                />
              </div>
            ))}
          </div>
        </div>

        {/* Spending Features */}
        <div className="mb-8">
          <h2 className="text-xl font-semibold text-gray-800 dark:text-white mb-4">
            Spending Patterns
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {spendingFeatures.map((feature) => (
              <div key={feature.name}>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  {feature.name}
                </label>
                <input
                  type="number"
                  name={feature.name}
                  value={formData[feature.name] || ''}
                  onChange={handleChange}
                  min={feature.min}
                  max={feature.max}
                  className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent bg-white dark:bg-gray-700 text-gray-800 dark:text-white"
                  required
                />
              </div>
            ))}
          </div>
        </div>

        {/* Purchase Features */}
        <div className="mb-8">
          <h2 className="text-xl font-semibold text-gray-800 dark:text-white mb-4">
            Purchase Behavior
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {purchaseFeatures.map((feature) => (
              <div key={feature.name}>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  {feature.name}
                </label>
                <input
                  type="number"
                  name={feature.name}
                  value={formData[feature.name] || ''}
                  onChange={handleChange}
                  min={feature.min}
                  max={feature.max}
                  className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent bg-white dark:bg-gray-700 text-gray-800 dark:text-white"
                  required
                />
              </div>
            ))}
          </div>
        </div>

        {/* Other Features */}
        {otherFeatures.length > 0 && (
          <div className="mb-8">
            <h2 className="text-xl font-semibold text-gray-800 dark:text-white mb-4">
              Additional Information
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {otherFeatures.map((feature) => (
                <div key={feature.name}>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    {feature.name}
                    {feature.description && (
                      <span className="text-xs text-gray-500 dark:text-gray-400 block">
                        {feature.description}
                      </span>
                    )}
                  </label>
                  <input
                    type="number"
                    name={feature.name}
                    value={formData[feature.name] || ''}
                    onChange={handleChange}
                    min={feature.min}
                    max={feature.max}
                    className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent bg-white dark:bg-gray-700 text-gray-800 dark:text-white"
                    required
                  />
                </div>
              ))}
            </div>
          </div>
        )}

        <div className="flex justify-end">
          <button
            type="submit"
            disabled={loading}
            className="px-6 py-3 bg-primary-500 text-white rounded-lg font-semibold hover:bg-primary-600 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {loading ? 'Predicting...' : '🔮 Predict Purchase Likelihood'}
          </button>
        </div>
      </form>
    </div>
  );
};

export default Prediction;

