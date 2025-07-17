/**
 * Entry point for XAI Trading Frontend
 */

import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import XAIApp from './App';

// Get the root element
const container = document.getElementById('root');
if (!container) {
  throw new Error('Root container missing in index.html');
}

// Create React root
const root = ReactDOM.createRoot(container);

// Render the app
root.render(
  <React.StrictMode>
    <XAIApp />
  </React.StrictMode>
);

// Enable hot module replacement in development
if (module.hot) {
  module.hot.accept();
}