/**
 * Ask AI Widget - Main Controller
 * 
 * This is the main entry point that integrates all modules:
 * - I18N: Internationalization
 * - API: Communication with backend
 * - UI: User interface management
 */

import { I18N, detectLanguage } from './ask-ai-i18n.js';
import { AskAIApi } from './ask-ai-api.js';
import { AskAIUI } from './ask-ai-ui.js';

class AskAIWidget {
  constructor() {
    // Check if marked library is loaded
    if (typeof marked === 'undefined') {
      console.error('Marked library not loaded!');
      console.info('Add this to your HTML: <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>');
      return;
    }

    // Initialize session and language
    this.sessionId = this.generateSessionId();
    this.language = detectLanguage();
    this.i18n = I18N[this.language.replace('-', '_')] || I18N.en;

    // Initialize modules
    this.api = new AskAIApi(this.sessionId, this.i18n);
    this.ui = new AskAIUI(this.i18n);

    // Configure marked
    marked.setOptions({
      breaks: true,
      gfm: true,
      headerIds: false,
      mangle: false,
    });

    this.init();
  }

  /**
   * Generate or retrieve session ID
   * @returns {string} Session ID
   */
  generateSessionId() {
    // Try to get existing session ID from sessionStorage
    let sessionId = sessionStorage.getItem('ask-ai-session-id');

    if (!sessionId) {
      // Generate new session ID if none exists
      sessionId = `browser-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
      sessionStorage.setItem('ask-ai-session-id', sessionId);
    }

    console.log('Using session ID:', sessionId);
    return sessionId;
  }

  /**
   * Initialize the widget
   */
  async init() {
    // Create UI
    this.ui.createWidget();
    
    // Bind events
    this.ui.bindEvents({
      onClose: () => this.ui.closeModal(),
      onClear: () => this.clearConversation(),
      onSend: () => this.sendMessage(),
      onThinkingToggle: () => this.ui.toggleThinking(),
    });

    // Bind feedback button events
    this.bindFeedbackEvents();

    // Observe theme changes
    this.ui.observeThemeChanges();

    // Check API connection
    await this.api.checkApiConnection();

    // Load conversation history
    await this.loadConversationHistory();

    // Add welcome message
    this.addWelcomeMessage();
  }

  /**
   * Bind feedback button events using event delegation
   */
  bindFeedbackEvents() {
    this.ui.messagesContainer.addEventListener('click', async (e) => {
      const target = e.target.closest('.ask-ai-feedback-btn');
      if (!target) return;

      const messageWrapper = target.closest('.ask-ai-message-wrapper');
      const messageDiv = messageWrapper?.querySelector('.ask-ai-message');
      const messageId = messageDiv?.getAttribute('data-message-id');
      
      if (!messageId) {
        console.warn('No message ID found for feedback');
        return;
      }

      const feedbackType = target.getAttribute('data-feedback');

      // Handle copy button
      if (target.classList.contains('copy')) {
        const feedbackDiv = target.closest('.ask-ai-feedback-actions');
        const content = feedbackDiv?.getAttribute('data-content') || '';
        await this.copyToClipboard(content);
        return;
      }

      // Handle like/dislike buttons
      if (feedbackType && messageId) {
        await this.handleFeedback(target, messageId, feedbackType);
        console.log('Submitting feedback with message ID:', messageId);
      }
    });
  }

  /**
   * Handle feedback submission
   * @param {HTMLElement} button - Feedback button element
   * @param {string} messageId - Message ID
   * @param {string} feedbackType - Feedback type ('like' or 'dislike')
   */
  async handleFeedback(button, messageId, feedbackType) {
    // Get all feedback buttons for this message
    const feedbackDiv = button.closest('.ask-ai-feedback-actions');
    const allButtons = feedbackDiv.querySelectorAll('.ask-ai-feedback-btn[data-feedback]');

    // Check if clicking the same button again (toggle off)
    if (button.classList.contains('active')) {
      button.classList.remove('active');
      return;
    }

    // Remove active state from all feedback buttons
    allButtons.forEach(btn => btn.classList.remove('active'));

    // Add active state to clicked button
    button.classList.add('active');

    // Submit feedback to backend
    const success = await this.api.submitFeedback(messageId, feedbackType);

    if (success) {
      console.log(`Feedback "${feedbackType}" submitted for message ${messageId}`);
    } else {
      console.error('Failed to submit feedback');
      // Remove active state on failure
      button.classList.remove('active');
    }
  }

  /**
   * Copy content to clipboard
   * @param {string} content - Content to copy
   */
  async copyToClipboard(content) {
    try {
      await navigator.clipboard.writeText(content);
      console.log('Content copied to clipboard');
      
      // Show a brief success message (optional)
      // You could add a toast notification here if desired
    } catch (error) {
      console.error('Failed to copy to clipboard:', error);
      
      // Fallback for older browsers
      const textArea = document.createElement('textarea');
      textArea.value = content;
      textArea.style.position = 'fixed';
      textArea.style.left = '-999999px';
      document.body.appendChild(textArea);
      textArea.select();
      try {
        document.execCommand('copy');
        console.log('Content copied to clipboard (fallback)');
      } catch (err) {
        console.error('Fallback copy failed:', err);
      }
      document.body.removeChild(textArea);
    }
  }

  /**
   * Load conversation history from API
   * Merges consecutive assistant messages into single messages
   */
  async loadConversationHistory() {
    const messages = await this.api.loadConversationHistory();

    if (messages && messages.length > 0) {
      // Group messages by conversation turn
      const turns = [];
      let currentTurn = null;
      
      for (let i = 0; i < messages.length; i++) {
        const msg = messages[i];
        
        if (!msg.content || typeof msg.content !== 'string') continue;
        
        const content = msg.content.trim();
        if (!content) continue;
        
        // Skip JSON array messages (tool calls - not rendered in history)
        let isJsonArray = false;
        try {
          const parsed = JSON.parse(content);
          isJsonArray = Array.isArray(parsed);
        } catch (e) {
          // Not valid JSON
        }
        if (isJsonArray) continue;
        
        if (msg.role === 'user') {
          currentTurn = {
            user: { content: content, id: msg.id },
            assistantTexts: []
          };
          turns.push(currentTurn);
        } else if (msg.role === 'assistant' && currentTurn) {
          currentTurn.assistantTexts.push(content);
        }
      }
      
      // Render each turn
      turns.forEach(turn => {
        // Render user message
        const userMessageId = turn.user.id || 'history_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
        this.ui.addMessage(turn.user.content, 'user', userMessageId);
        
        // Render merged assistant message
        if (turn.assistantTexts.length > 0) {
          const mergedText = turn.assistantTexts.join('\n\n');
          const assistantMessageId = 'history_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
          const contentWithSuffix = mergedText + (this.i18n.helpSuffix || '');
          this.ui.addMessage(contentWithSuffix, 'assistant', assistantMessageId);
        }
      });

      if (turns.length > 0) {
        this.ui.scrollToBottom();
      }
    }
  }

  /**
   * Add welcome message based on API connection status
   */
  addWelcomeMessage() {
    this.ui.addWelcomeMessage(this.api.apiConnected);
  }

  /**
   * Send user message
   */
  async sendMessage() {
    const message = this.ui.getInputValue().trim();
    if (!message || this.ui.isTyping) return;

    // Add user message to UI
    this.ui.addMessage(message, 'user');
    this.ui.clearInput();

    // Create assistant message placeholder
    const assistantMessageDiv = this.ui.showTypingIndicator();
    const messageId = assistantMessageDiv.getAttribute('data-message-id') || 
                      `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    assistantMessageDiv.setAttribute('data-message-id', messageId);

    // Track active tool calls by call_id
    const activeToolCalls = new Map();
    // Track the currently active thinking container
    let activeThinkingContainer = null;

    // Build model_config based on thinking toggle state
    const modelConfig = {
      enable_thinking: this.ui.enableThinking
    };

    try {
      // Get AI response with streaming
      await this.api.getAIResponseStream(
        message,
        // onContentUpdate
        (content) => {
          // Remove typing class when we start receiving content
          assistantMessageDiv.classList.remove('typing');
          // Finalize thinking panel if it was open (thinking phase ended, text phase started)
          if (activeThinkingContainer) {
            this.ui.finalizeThinking(activeThinkingContainer);
            activeThinkingContainer = null;
          }
          // Collapse completed tool containers when text content arrives
          this.ui.collapseCompletedToolContainers(assistantMessageDiv);
          this.ui.updateMessageContent(assistantMessageDiv, content);
        },
        // onToolUse
        (toolName, toolArgs, callId) => {
          // Remove typing indicator when first tool is called
          assistantMessageDiv.classList.remove('typing');
          // Clear the typing dots content
          const typingIndicator = assistantMessageDiv.querySelector('.typing-indicator');
          if (typingIndicator) {
            typingIndicator.remove();
          }

          // Finalize thinking panel if it was open before tool call
          if (activeThinkingContainer) {
            this.ui.finalizeThinking(activeThinkingContainer);
            activeThinkingContainer = null;
          }
          
          // Add tool call to panel
          const toolId = this.ui.addToolCall(toolName, toolArgs, assistantMessageDiv);
          
          // Store mapping from callId to toolId
          if (callId) {
            activeToolCalls.set(callId, toolId);
          }
        },
        // onComplete
        (userMessage, assistantMessage) => {
          // Mark all remaining tools as done
          activeToolCalls.forEach(toolId => {
            this.ui.markToolCallDone(toolId);
          });

          // Finalize any remaining active thinking container
          if (activeThinkingContainer) {
            this.ui.finalizeThinking(activeThinkingContainer);
            activeThinkingContainer = null;
          }
          
          // Ensure typing class is removed
          assistantMessageDiv.classList.remove('typing');
          // Remove the typingIndicator ID to prevent it from being deleted
          assistantMessageDiv.removeAttribute('id');
          
          // Extract content from assistantMessage
          let finalContent = '';
          if (typeof assistantMessage.content === 'string') {
            finalContent = assistantMessage.content;
          } else if (Array.isArray(assistantMessage.content)) {
            finalContent = assistantMessage.content
              .filter(c => c.type === "text")
              .map(c => c.text)
              .join('');
          }
          
          // Update with verified content and add helpSuffix at the end
          this.ui.finalizeMessage(assistantMessageDiv, finalContent);
          
          // Update with server-provided message ID
          if (assistantMessage.id) {
            assistantMessageDiv.setAttribute('data-message-id', assistantMessage.id);
          }

          const messageWrapper = assistantMessageDiv.parentNode;
          const hasFeedbackButtons = messageWrapper?.classList.contains('ask-ai-message-wrapper') && 
                                    messageWrapper.querySelector('.ask-ai-feedback-actions');
          
          if (!hasFeedbackButtons && assistantMessage.id) {
            console.log('Adding feedback buttons in onComplete');
            this.ui.addFeedbackButtons(assistantMessageDiv, assistantMessage.id, finalContent);
          } else if (hasFeedbackButtons) {
            // Update stored content for copying
            const feedbackDiv = messageWrapper.querySelector('.ask-ai-feedback-actions');
            if (feedbackDiv) {
              feedbackDiv.setAttribute('data-content', finalContent);
            }
          }
          
          // Mark typing as complete
          this.ui.isTyping = false;
          this.ui.sendBtn.disabled = false;
          
          // Store message with server ID
          this.ui.messages.push({
            content: finalContent,
            type: 'assistant',
            timestamp: Date.now(),
            messageId: assistantMessage.id
          });
        },
        // onError
        (error) => {
          assistantMessageDiv.classList.remove('typing');
          assistantMessageDiv.removeAttribute('id');
          this.ui.updateMessageContent(assistantMessageDiv, this.i18n.connectionError);
          this.ui.isTyping = false;
          this.ui.sendBtn.disabled = false;
          console.error('AI response error:', error);
        },
        // onToolComplete
        (callId) => {
          // Mark specific tool as done when its output is received
          const toolId = activeToolCalls.get(callId);
          if (toolId) {
            this.ui.markToolCallDone(toolId);
            console.log('Tool completed:', callId, '-> toolId:', toolId);
          }
        },
        // modelConfig
        modelConfig,
        // onThinkingUpdate
        (thinkingText) => {
          // Remove typing class when we start receiving thinking content
          assistantMessageDiv.classList.remove('typing');
          const typingIndicator = assistantMessageDiv.querySelector('.typing-indicator');
          if (typingIndicator) {
            typingIndicator.remove();
          }
          // Collapse completed tool containers when thinking phase starts
          this.ui.collapseCompletedToolContainers(assistantMessageDiv);
          // Create a new thinking container if none is active
          if (!activeThinkingContainer) {
            activeThinkingContainer = this.ui.createThinkingContainer(assistantMessageDiv);
          }
          this.ui.appendThinkingContent(thinkingText, activeThinkingContainer);
        }
      );
    } catch (error) {
      assistantMessageDiv.classList.remove('typing');
      assistantMessageDiv.removeAttribute('id');
      this.ui.updateMessageContent(assistantMessageDiv, this.i18n.sendError);
      this.ui.isTyping = false;
      this.ui.sendBtn.disabled = false;
      console.error('Send message error:', error);
    }
  }


  /**
   * Clear conversation history
   */
  async clearConversation() {
    if (!confirm(this.i18n.clearConfirm)) {
      return;
    }

    const success = await this.api.clearConversation();

    if (success) {
      // Clear UI
      this.ui.clearMessages();

      // Add welcome message back
      this.addWelcomeMessage();

      console.log('Conversation cleared successfully');
    } else {
      alert(this.i18n.clearFailed);
    }
  }
}

// Initialize the widget when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
  new AskAIWidget();
});

// Export for potential external usage
if (typeof module !== 'undefined' && module.exports) {
  module.exports = AskAIWidget;
}

export default AskAIWidget;