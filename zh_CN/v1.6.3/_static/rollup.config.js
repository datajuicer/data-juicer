/**
 * Rollup Configuration for Ask AI Widget
 * 
 * This configuration bundles the modular JavaScript files into a single file
 * that can be used directly in Sphinx documentation without module system support.
 */

import resolve from '@rollup/plugin-node-resolve';

export default {
  input: 'ask-ai-modules/ask-ai-widget.js',
  output: {
    file: 'ask-ai-widget.js',
    format: 'iife',
    name: 'AskAIWidget',
    banner: '/**\n * Ask AI Widget - Bundled Version\n * Generated from modular source files\n */',
    globals: {
      marked: 'marked'
    }
  },
  external: ['marked'],
  plugins: [
    resolve()
  ]
};
