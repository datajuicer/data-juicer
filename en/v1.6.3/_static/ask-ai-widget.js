/**
 * Ask AI Widget - Bundled Version
 * Generated from modular source files
 */
var AskAIWidget = (function () {
  'use strict';

  /**
   * Ask AI Widget - Internationalization Module
   */

  const I18N = {
    en: {
      title: 'Data-Juicer Q&A Copilot [Beta]',
      buttonTitle: 'Ask Juicer',
      clearTitle: 'Restart conversation',
      expandTitle: 'Expand/Collapse',
      collapseTitle: 'Collapse',
      minimizeTitle: 'Minimize',
      sendTitle: 'Send message',
      closeTitle: 'Close panel',
      inputPlaceholder: 'Type your question here...',
      barPlaceholder: 'Ask Juicer anything about this project...',
      askSelection: 'Ask AI',
      disclaimer: 'Results are AI-generated and for reference only.',
      welcomeMessage: '👋 Hi! I\'m Juicer. Ask me anything about Data-Juicer!<br><br><small style="color: #888;">Powered by <a href="https://github.com/datajuicer/data-juicer-agents" target="_blank" style="color: #667eea; text-decoration: none;">data-juicer-agents</a> · Results are AI-generated and for reference only.</small>',
      welcomeConnected: '👋 Hi! I\'m Juicer. <span style="color: #28a745;">🟢 Connected</span><br>Ask me anything about Data-Juicer!<br><br><small style="color: #888;">Powered by <a href="https://github.com/datajuicer/data-juicer-agents" target="_blank" style="color: #667eea; text-decoration: none;">data-juicer-agents</a> · Results are AI-generated and for reference only.</small>',
      welcomeOffline: '👋 Hi! I\'m Juicer. <span style="color: #dc3545;">🔴 Offline Mode</span><br>Please ensure the API service is running.<br><br><small style="color: #888;">Powered by <a href="https://github.com/datajuicer/data-juicer-agents" target="_blank" style="color: #667eea; text-decoration: none;">data-juicer-agents</a> · Results are AI-generated and for reference only.</small>',
      clearConfirm: 'Are you sure you want to clear the conversation history? This action cannot be undone.',
      clearFailed: 'Failed to clear conversation history. Please try again.',
      clearError: 'Error clearing conversation history. Please check your connection and try again.',
      sendError: 'Sorry, I encountered an error. Please try again later.',
      noResponse: 'I processed your request but no valid response was generated. Please try again.',
      connectionError: 'Unable to connect to AI service, please check network or contact administrator.',
      offlineResponse: 'Sorry, the Q&A Bot API is not configured or currently unavailable. Please refer to the Data-Juicer documentation for information, or contact the administrator to configure the API service.',
      usingTool: 'Using',
      toolCalls: 'Tool Calls',
      done: 'Done',
      running: 'Running',
      like: 'Like',
      dislike: 'Dislike',
      copyMarkdown: 'Copy Markdown',
      feedbackSuccess: 'Thank you for your feedback!',
      feedbackError: 'Failed to submit feedback',
      copiedSuccess: 'Copied to clipboard!',
      thinking: 'Thinking',
      thinkingTitle: 'Enable/Disable Thinking',
      thinkingContent: 'Thinking',
      helpSuffix: '\n\n---\n*If you have any questions, please visit [data-juicer issues](https://github.com/datajuicer/data-juicer/issues) or [data-juicer-agents issues](https://github.com/datajuicer/data-juicer-agents/issues)*'
    },
    zh_CN: {
      title: 'Data-Juicer Q&A Copilot [Beta]',
      buttonTitle: '询问 Juicer',
      clearTitle: '重新开始对话',
      expandTitle: '展开/收起',
      collapseTitle: '收起',
      minimizeTitle: '最小化',
      sendTitle: '发送消息',
      closeTitle: '关闭面板',
      inputPlaceholder: '在此输入您的问题...',
      barPlaceholder: '关于本项目有任何问题，问问 Juicer...',
      askSelection: '问问 AI',
      disclaimer: '结果由 AI 生成，仅供参考。',
      welcomeMessage: '👋 你好！我是 Juicer。问我任何关于 Data-Juicer 的问题！<br><br><small style="color: #888;">技术支持：<a href="https://github.com/datajuicer/data-juicer-agents" target="_blank" style="color: #667eea; text-decoration: none;">data-juicer-agents</a> · 结果由 AI 生成，仅供参考。</small>',
      welcomeConnected: '👋 你好！我是 Juicer。<span style="color: #28a745;">🟢 已连接</span><br>问我任何关于 Data-Juicer 的问题！<br><br><small style="color: #888;">技术支持：<a href="https://github.com/datajuicer/data-juicer-agents" target="_blank" style="color: #667eea; text-decoration: none;">data-juicer-agents</a> · 结果由 AI 生成，仅供参考。</small>',
      welcomeOffline: '👋 你好！我是 Juicer。<span style="color: #dc3545;">🔴 离线模式</span><br>请确保 API 服务正在运行。<br><br><small style="color: #888;">技术支持：<a href="https://github.com/datajuicer/data-juicer-agents" target="_blank" style="color: #667eea; text-decoration: none;">data-juicer-agents</a> · 结果由 AI 生成，仅供参考。</small>',
      clearConfirm: '确定要清除对话历史吗？此操作无法撤销。',
      clearFailed: '清除对话历史失败。请重试。',
      clearError: '清除对话历史时出错。请检查您的连接并重试。',
      sendError: '抱歉，遇到了一个错误。请稍后再试。',
      noResponse: '我处理了您的请求，但没有生成有效的响应。请重试。',
      connectionError: '无法连接到 AI 服务，请检查网络或联系管理员。',
      offlineResponse: '抱歉，Juicer API 未配置或当前不可用。请参阅 Data-Juicer 文档获取信息，或联系管理员配置 API 服务。',
      usingTool: '正在调用',
      toolCalls: '工具调用',
      done: '完成',
      running: '执行中',
      like: '点赞',
      dislike: '点踩',
      copyMarkdown: '复制 Markdown',
      feedbackSuccess: '感谢您的反馈！',
      feedbackError: '提交反馈失败',
      copiedSuccess: '已复制到剪贴板！',
      thinking: '思考',
      thinkingTitle: '开启/关闭思考模式',
      thinkingContent: '思考',
      helpSuffix: '\n\n---\n*如果您有任何问题，请访问 [data-juicer issues](https://github.com/datajuicer/data-juicer/issues) 或 [data-juicer-agents issues](https://github.com/datajuicer/data-juicer-agents/issues)*'
    }
  };

  /**
   * Detect the current language from HTML attributes or URL
   * @returns {string} Language code (e.g., 'en' or 'zh_CN')
   */
  function detectLanguage() {
    // Try to get language from HTML lang attribute
    const htmlLang = document.documentElement.lang;
    if (htmlLang) {
      // Handle both 'zh-CN' and 'zh_CN' formats
      const normalizedLang = htmlLang.replace('-', '_');
      if (I18N[normalizedLang]) {
        return normalizedLang;
      }
      // Try just the language code (e.g., 'zh' from 'zh-CN')
      const langCode = normalizedLang.split('_')[0];
      if (langCode === 'zh') {
        return 'zh_CN';
      }
    }
    
    // Try to get language from meta tag
    const metaLang = document.querySelector('meta[name="language"]');
    if (metaLang && metaLang.content) {
      const normalizedLang = metaLang.content.replace('-', '_');
      if (I18N[normalizedLang]) {
        return normalizedLang;
      }
    }
    
    // Try to detect from URL (e.g., /zh_CN/version/page.html)
    const urlMatch = window.location.pathname.match(/\/(zh_CN|en)\//);
    if (urlMatch && I18N[urlMatch[1]]) {
      return urlMatch[1];
    }
    
    // Default to English
    return 'en';
  }

  /**
   * Ask AI Widget - API Communication Layer
   */

  class AskAIApi {
    constructor(sessionId, i18n) {
      this.sessionId = sessionId;
      this.i18n = i18n;
      this.apiConnected = false;
    }

    getApiBaseUrl() {
      const metaApiUrl = document.querySelector('meta[name="juicer-api-url"]');
      if (metaApiUrl && metaApiUrl.content) {
        return metaApiUrl.content;
      }

      if (window.JUICER_API_URL) {
        return window.JUICER_API_URL;
      }

      return 'http://localhost:8080';
    }

    async checkApiConnection() {
      try {
        const response = await fetch(`${this.getApiBaseUrl()}/health`, {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json',
          }
        });

        if (response.ok) {
          const data = await response.json();
          this.apiConnected = data.status === 'healthy';
        } else {
          this.apiConnected = false;
        }
      } catch (error) {
        console.warn('Failed to connect to API:', error);
        this.apiConnected = false;
      }
      return this.apiConnected;
    }

    /**
     * Get the latest messages from server memory
     * @param {number} limit - Number of recent messages to fetch (default: 10)
     * @returns {Promise<Array>} Array of messages with complete metadata
     */
    async getMemory(limit = 10) {
      if (!this.apiConnected) {
        console.log('API not connected, skipping memory fetch');
        return [];
      }

      try {
        const requestBody = {
          input: [
            {
              role: "user",
              content: [
                {
                  type: "text",
                  text: "",
                },
              ],
            },
          ],
          session_id: this.sessionId,
          user_id: this.sessionId,
        };

        console.log('Fetching memory for session:', this.sessionId);

        const response = await fetch(`${this.getApiBaseUrl()}/memory`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify(requestBody)
        });

        if (response.ok) {
          const data = await response.json();
          const messages = data.messages || [];
          console.log('Memory fetched:', messages.length, 'messages');

          // Return latest messages if limit is specified
          return limit > 0 ? messages.slice(-limit) : messages;
        } else {
          console.warn('Failed to fetch memory:', response.status);
          return [];
        }
      } catch (error) {
        console.warn('Error fetching memory:', error);
        return [];
      }
    }

    /**
     * Load conversation history from server
     * @returns {Promise<Array>} Array of historical messages
     */
    async loadConversationHistory() {
      const messages = await this.getMemory(0); // Get all messages
      console.log('Loaded', messages.length, 'historical messages');
      return messages;
    }

    async clearConversation() {
      try {
        const requestBody = {
          input: [
            {
              role: "user",
              content: [
                {
                  type: "text",
                  text: "",
                },
              ],
            },
          ],
          session_id: this.sessionId,
          user_id: this.sessionId,
        };

        const response = await fetch(`${this.getApiBaseUrl()}/clear`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'x-session-id': this.sessionId
          },
          body: JSON.stringify(requestBody)
        });

        if (response.ok) {
          console.log('Conversation history cleared successfully');
          return true;
        } else {
          console.error('Failed to clear conversation history:', response.status);
          return false;
        }
      } catch (error) {
        console.error('Error clearing conversation history:', error);
        return false;
      }
    }

    /**
     * Get AI response using streaming, then sync with server memory
     * @param {string} message - User message
     * @param {Function} onContentUpdate - Callback for content updates (text)
     * @param {Function} onToolUse - Callback for tool usage (toolName, toolArgs, callId)
     * @param {Function} onComplete - Callback when complete with verified messages (userMessage, assistantMessage)
     * @param {Function} onError - Callback for errors (error)
     * @param {Function} onToolComplete - Callback when tool execution completes (callId)
     * @param {Object} modelConfig - Optional model configuration (e.g., { enable_thinking: true })
     * @param {Function} onThinkingUpdate - Callback for thinking content updates (thinkingText)
     */
    async getAIResponseStream(message, onContentUpdate, onToolUse, onComplete, onError, onToolComplete, modelConfig = null, onThinkingUpdate = null) {
      let currentStreamContent = '';
      let streamMessageId = `temp_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      let hasReceivedContent = false;
      let streamCompletedSuccessfully = false;
      let isInReasoningPhase = false;

      try {
        const requestBody = {
          input: [
            {
              role: "user",
              content: [
                {
                  type: "text",
                  text: message.trim(),
                },
              ],
            },
          ],
          session_id: this.sessionId,
          user_id: this.sessionId,
        };

        if (modelConfig && typeof modelConfig === 'object') {
          requestBody.model_params = modelConfig;
        }

        console.log('Sending streaming request to:', `${this.getApiBaseUrl()}/process`);

        const response = await fetch(`${this.getApiBaseUrl()}/process`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'x-session-id': this.sessionId,
          },
          body: JSON.stringify(requestBody),
        });

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        // Process stream
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop() || '';

          for (const line of lines) {
            if (!line.trim() || !line.startsWith('data:')) continue;

            const jsonString = line.slice(5).trim();
            if (!jsonString) continue;

            try {
              const data = JSON.parse(jsonString);

              // End of stream
              if (data.object === "response" && data.status === "completed") {
                console.log('Stream ended normally.');
                streamCompletedSuccessfully = true;
                break;
              }

              // Handle errors
              if (data.object === "response" && data.error) {
                throw new Error(data.error.message || 'An error occurred during processing.');
              }

              // Handle tool use
              if (data.object === "message" && data.type === "plugin_call") {
                if (Array.isArray(data.content)) {
                  const toolCallWithId = data.content[0]?.type === "data" ? data.content[0].data : null;
                  const toolCallWithArgs = data.content.length > 1 ? data.content[1] : data.content[0];
                  const toolCall = toolCallWithArgs?.type === "data" ? toolCallWithArgs.data : null;

                  if (toolCall && onToolUse) {
                    const toolName = toolCall.name || 'Unknown Tool';
                    let toolArgs = toolCall.arguments || {};

                    if (typeof toolArgs === 'string') {
                      try {
                        toolArgs = JSON.parse(toolArgs);
                      } catch (e) {
                        console.warn('Failed to parse tool arguments:', e);
                        toolArgs = {};
                      }
                    }

                    const callId = toolCallWithId?.call_id || null;
                    console.log('Tool call detected:', { toolName, toolArgs, callId });
                    onToolUse(toolName, toolArgs, callId);
                  }
                }
              }

              // Handle tool output
              if (data.object === "message" && data.type === "plugin_call_output") {
                if (Array.isArray(data.content)) {
                  const outputData = data.content.find(item => item.type === "data")?.data;
                  const callId = outputData?.call_id || null;
                  const output = outputData?.output;

                  if (output && callId && onToolComplete) {
                    console.log('Tool output received for call_id:', callId);
                    onToolComplete(callId);
                  }
                }
              }

              // Handle reasoning phase: object="message", type="reasoning"
              if (data.object === "message" && data.type === "reasoning") {
                if (data.status === "in_progress") {
                  isInReasoningPhase = true;
                  console.log('Entering reasoning phase, msg_id:', data.id);
                } else if (data.status === "completed") {
                  // Reasoning phase completed - ignore entirely (skip all further processing)
                  isInReasoningPhase = false;
                  console.log('Reasoning phase completed, msg_id:', data.id);
                  continue;
                }
              }

              // Handle normal message phase start: object="message", type="message"
              // This signals the end of reasoning and start of normal response
              if (data.object === "message" && data.type === "message" && data.status === "in_progress") {
                if (isInReasoningPhase) {
                  isInReasoningPhase = false;
                  console.log('Exiting reasoning phase, entering message phase');
                }
              }

              // Skip non-delta content events (e.g., completed content summaries from reasoning)
              if (
                data.object === "content" &&
                data.type === "text" &&
                data.delta !== true &&
                data.status === "completed"
              ) {
                console.log('Skipping completed content summary, sequence:', data.sequence_number);
                continue;
              }

              // Handle incremental text content (used for both reasoning and normal text)
              if (
                data.object === "content" &&
                data.type === "text" &&
                data.delta === true &&
                data.text !== undefined
              ) {
                if (isInReasoningPhase) {
                  // Route to thinking callback during reasoning phase
                  if (onThinkingUpdate) {
                    onThinkingUpdate(data.text);
                  }
                } else {
                  // Route to normal content callback
                  if (!hasReceivedContent) {
                    currentStreamContent = '';
                    hasReceivedContent = true;
                  }
                  currentStreamContent += data.text;
                  if (onContentUpdate) {
                    onContentUpdate(currentStreamContent);
                  }
                }
              }

              // Final message delivery (skip reasoning messages to avoid showing thinking as normal text)
              if (
                data.object === "message" &&
                data.status === "completed" &&
                data.role === "assistant" &&
                data.type !== "reasoning" &&
                Array.isArray(data.content)
              ) {
                const fullText = data.content
                  .filter(c => c.type === "text")
                  .map(c => c.text)
                  .join('');

                if (fullText && !hasReceivedContent) {
                  currentStreamContent = fullText;
                  hasReceivedContent = true;
                  if (onContentUpdate) {
                    onContentUpdate(currentStreamContent);
                  }
                }

                if (data.id) {
                  streamMessageId = data.id;
                  console.log('Received message ID from stream:', data.id);
                }
              }
            } catch (parseError) {
              console.warn('Failed to parse SSE data:', parseError);
            }
          }
        }

        // Validate stream completion
        if (!hasReceivedContent || !currentStreamContent.trim()) {
          console.warn('Stream completed but no content received, will fetch from memory');
          currentStreamContent = this.i18n.noResponse;
        }

        // ✨ Fetch from memory to get verified messages with complete metadata
        console.log('Stream ended, fetching latest messages from memory...');
        const recentMessages = await this.getMemory(10); // Get more messages to debug

        // Find the last user message and last assistant message
        const extractText = (content) => {
          if (Array.isArray(content)) {
            return content.filter(c => c.type === 'text').map(c => c.text).join('').trim();
          }
          return (content || '').trim();
        };

        // Get the last user message and last assistant message from memory
        let lastUserMessage = null;
        let lastAssistantMessage = null;

        for (let i = recentMessages.length - 1; i >= 0; i--) {
          const msg = recentMessages[i];
          if (!lastAssistantMessage && msg.role === 'assistant') {
            lastAssistantMessage = msg;
          }
          if (!lastUserMessage && msg.role === 'user') {
            lastUserMessage = msg;
          }
          if (lastUserMessage && lastAssistantMessage) break;
        }

        const expectedContent = message.trim();
        const lastUserContent = lastUserMessage ? extractText(lastUserMessage.content) : '';

        if (lastUserMessage && lastAssistantMessage && lastUserContent === expectedContent) {
          const assistantContent = extractText(lastAssistantMessage.content);

          console.log('✓ Memory sync successful');
          console.log('  User message ID:', lastUserMessage.id);
          console.log('  Assistant message ID:', lastAssistantMessage.id);

          // Use server content if stream was incomplete or content differs
          if (assistantContent && (!streamCompletedSuccessfully)) {
            console.log('⚠ Stream content differs from server, using server version');
            currentStreamContent = assistantContent;
            if (onContentUpdate) {
              onContentUpdate(currentStreamContent);
            }
          }

          if (onComplete) {
            onComplete(lastUserMessage, lastAssistantMessage);
          }
          return;
        } else {
          console.warn('⚠ Could not verify messages from memory');
          console.log('  Expected user content:', expectedContent.substring(0, 50));
          console.log('  Found user content:', lastUserContent.substring(0, 50));
        }

        // Fallback: memory sync failed, use stream data
        console.warn('⚠ Could not verify messages from memory, using stream data');
        if (onComplete) {
          // Create message objects from stream data
          const fallbackUserMessage = {
            id: 'user_' + streamMessageId,
            role: 'user',
            content: [{ type: 'text', text: message.trim() }]
          };
          const fallbackAssistantMessage = {
            id: streamMessageId,
            role: 'assistant',
            content: [{ type: 'text', text: currentStreamContent }]
          };
          onComplete(fallbackUserMessage, fallbackAssistantMessage);
        }

      } catch (error) {
        console.error('Stream error:', error);
        if (onError) {
          onError(error);
        }
      }
    }

    getOfflineResponse(message) {
      return this.i18n.offlineResponse;
    }

    async submitFeedback(messageId, feedbackType, comment = '') {
      try {
        const requestBody = {
          session_id: this.sessionId,
          user_id: this.sessionId,
          data: {
            feedback_type: feedbackType,
            message_id: messageId,
            comment: comment
          }
        };

        console.log('Submitting feedback:', requestBody);

        const response = await fetch(`${this.getApiBaseUrl()}/feedback`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'x-session-id': this.sessionId
          },
          body: JSON.stringify(requestBody)
        });

        if (response.ok) {
          const data = await response.json();
          console.log('Feedback submitted successfully:', data);
          return data.status === 'ok';
        } else {
          console.error('Failed to submit feedback:', response.status);
          return false;
        }
      } catch (error) {
        console.error('Error submitting feedback:', error);
        return false;
      }
    }
  }

  /**
   * Ask AI Widget - UI Management Module
   *
   * UI: bottom floating AI bar, selection tooltip and a
   * drag-to-resize side panel, integrated with the real Q&A capabilities
   * (streaming responses, thinking mode, tool calls, feedback).
   */

  const SPARKLE_ICON = '<svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M5 2.5L3.5 2L3 0.5 2.5 2 1 2.5l1.5.5L3 4.5 3.5 3 5 2.5z" fill="currentColor"/><path d="M8 2l1.5 3.5L13 7l-3.5 1.5L8 12l-1.5-3.5L3 7l3.5-1.5L8 2z" stroke="currentColor" stroke-width="1.2" stroke-linecap="round" stroke-linejoin="round"/></svg>';
  const SEND_ICON = '<svg width="14" height="14" viewBox="0 0 16 16" fill="none"><path d="M2 8l12-6-4 6 4 6L2 8z" fill="currentColor"/></svg>';
  const TRASH_ICON = '<svg width="15" height="15" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M2.5 4.5h11M5.5 4.5V3a1 1 0 011-1h3a1 1 0 011 1v1.5M6.5 7v4.5M9.5 7v4.5"/><path d="M3.5 4.5l.5 8.5a1 1 0 001 1h6a1 1 0 001-1l.5-8.5"/></svg>';
  const CLOSE_ICON = '<svg width="15" height="15" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"><path d="M4 4l8 8M12 4l-8 8"/></svg>';
  const BRAIN_ICON = '<svg width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.3" stroke-linecap="round" stroke-linejoin="round"><path d="M8 2.5c-1 0-1.8.6-2.1 1.4C4.6 4 3.5 5 3.5 6.4c0 .8.3 1.4.8 1.9-.3.4-.5 1-.5 1.6 0 1.5 1.2 2.6 2.7 2.6.6 0 1.1-.2 1.5-.5.4.3.9.5 1.5.5 1.5 0 2.7-1.1 2.7-2.6 0-.6-.2-1.2-.5-1.6.5-.5.8-1.1.8-1.9 0-1.4-1.1-2.4-2.4-2.5C9.8 3.1 9 2.5 8 2.5z"/><path d="M8 2.5V13"/></svg>';
  const LIKE_ICON = '<svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.3" stroke-linecap="round" stroke-linejoin="round"><path d="M5 7v6.5H3a1 1 0 01-1-1V8a1 1 0 011-1h2zm0 0l2.2-4.2a1.5 1.5 0 012.8.7V6h3.2a1.5 1.5 0 011.5 1.8l-1 4.5a1.5 1.5 0 01-1.5 1.2H5"/></svg>';
  const DISLIKE_ICON = '<svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.3" stroke-linecap="round" stroke-linejoin="round"><path d="M11 9V2.5H13a1 1 0 011 1V8a1 1 0 01-1 1h-2zm0 0l-2.2 4.2a1.5 1.5 0 01-2.8-.7V10H2.8a1.5 1.5 0 01-1.5-1.8l1-4.5A1.5 1.5 0 013.8 2.5H11"/></svg>';
  const COPY_ICON = '<svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.3" stroke-linecap="round" stroke-linejoin="round"><rect x="5.5" y="5.5" width="8" height="8" rx="1.5"/><path d="M3 10.5V3.5a1 1 0 011-1h7"/></svg>';

  const PANEL_WIDTH_KEY = 'ask-ai-panel-width';
  const PANEL_MIN_WIDTH = 320;
  const PANEL_MAX_WIDTH = 720;
  const PANEL_DEFAULT_WIDTH = 420;

  class AskAIUI {
    constructor(i18n) {
      this.i18n = i18n;
      this.isOpen = false;
      this.isExpanded = false;
      this.isTyping = false;
      this.enableThinking = false;
      this.messages = [];
      this.contextChips = [];

      // DOM references (will be set after createWidget)
      this.root = null;
      this.bar = null;
      this.barInput = null;
      this.barSendBtn = null;
      this.panel = null;
      this.closeBtn = null;
      this.clearBtn = null;
      this.thinkingBtn = null;
      this.messagesContainer = null;
      this.input = null;
      this.sendBtn = null;
      this.chipsRow = null;
      this.selectionTooltip = null;
    }

    /**
     * Create the widget HTML structure
     */
    createWidget() {
      const root = document.createElement('div');
      root.className = 'ask-ai-widget';
      root.innerHTML = `
      <!-- Bottom floating AI bar -->
      <div class="ai-assistant-bar" id="askAiBar">
        <div class="ai-input-wrapper">
          <span class="ai-input-icon">${SPARKLE_ICON}</span>
          <input type="text" class="ai-input" id="askAiBarInput"
            placeholder="${this.i18n.barPlaceholder}" autocomplete="off" />
          <button class="ai-send-btn" id="askAiBarSend" aria-label="${this.i18n.sendTitle}" title="${this.i18n.sendTitle}">
            ${SEND_ICON}
          </button>
        </div>
      </div>

      <!-- Selection tooltip -->
      <div class="ai-selection-tooltip" id="askAiSelectionTooltip">
        ${SPARKLE_ICON}
        ${this.i18n.askSelection}
      </div>

      <!-- Side panel -->
      <div class="ai-panel" id="askAiPanel">
        <div class="ai-panel-drag" id="askAiPanelDrag"></div>
        <div class="ai-panel-header">
          <div class="ai-panel-header-left">
            <span class="ai-panel-icon">${SPARKLE_ICON}</span>
            <span class="ai-panel-title">${this.i18n.title}</span>
          </div>
          <div class="ai-panel-header-actions">
            <button class="ask-ai-thinking-toggle" id="askAiThinkingToggle" title="${this.i18n.thinkingTitle}">
              ${BRAIN_ICON}
              <span class="ask-ai-thinking-label">${this.i18n.thinking}</span>
            </button>
            <button class="ai-panel-action-btn" id="askAiClear" aria-label="${this.i18n.clearTitle}" title="${this.i18n.clearTitle}">
              ${TRASH_ICON}
            </button>
            <button class="ai-panel-action-btn" id="askAiClose" aria-label="${this.i18n.closeTitle}" title="${this.i18n.closeTitle}">
              ${CLOSE_ICON}
            </button>
          </div>
        </div>

        <div class="ai-panel-messages ask-ai-messages" id="askAiMessages">
          <div class="ask-ai-welcome">
            ${this.i18n.welcomeMessage}
          </div>
        </div>

        <div class="ai-panel-input">
          <div class="ai-panel-chips-row" id="askAiChips"></div>
          <div class="ai-panel-input-wrapper">
            <textarea
              class="ask-ai-panel-input"
              id="askAiInput"
              placeholder="${this.i18n.inputPlaceholder}"
              rows="1"
            ></textarea>
            <button class="ai-send-btn active" id="askAiSend" aria-label="${this.i18n.sendTitle}" title="${this.i18n.sendTitle}">
              ${SEND_ICON}
            </button>
          </div>
        </div>
        <div class="ai-panel-disclaimer">${this.i18n.disclaimer}</div>
      </div>
    `;

      document.body.appendChild(root);

      // Store references
      this.root = root;
      this.bar = document.getElementById('askAiBar');
      this.barInput = document.getElementById('askAiBarInput');
      this.barSendBtn = document.getElementById('askAiBarSend');
      this.panel = document.getElementById('askAiPanel');
      this.closeBtn = document.getElementById('askAiClose');
      this.clearBtn = document.getElementById('askAiClear');
      this.thinkingBtn = document.getElementById('askAiThinkingToggle');
      this.messagesContainer = document.getElementById('askAiMessages');
      this.input = document.getElementById('askAiInput');
      this.sendBtn = document.getElementById('askAiSend');
      this.chipsRow = document.getElementById('askAiChips');
      this.selectionTooltip = document.getElementById('askAiSelectionTooltip');

      // Restore persisted panel width
      const savedWidth = parseInt(localStorage.getItem(PANEL_WIDTH_KEY) || '', 10);
      if (savedWidth && !Number.isNaN(savedWidth)) {
        document.body.style.setProperty('--ai-panel-width', `${savedWidth}px`);
      }
    }

    /**
     * Attach an Enter-to-send handler with IME composition guards.
     *
     * IME composition guard: On macOS, when a user confirms an English
     * candidate (e.g. "json") by pressing Enter under a Chinese IME,
     * some browsers fire `compositionend` BEFORE `keydown(Enter)`,
     * causing all standard guards (e.isComposing, keyCode 229) to fail.
     *
     * Solution: record the timestamp of the last `compositionend` and
     * suppress any Enter keydown that arrives within a short window
     * after it — that Enter was used to confirm the IME candidate,
     * not to send the message.
     * @param {HTMLElement} inputEl - Input element
     * @param {Function} handler - Send handler
     */
    _attachEnterToSend(inputEl, handler) {
      let lastCompositionEndTime = 0;
      inputEl.addEventListener('compositionstart', () => {
        lastCompositionEndTime = 0;
      });
      inputEl.addEventListener('compositionend', () => {
        lastCompositionEndTime = Date.now();
      });

      inputEl.addEventListener('keydown', (e) => {
        if (e.key !== 'Enter' || e.shiftKey) return;

        // Standard IME guards
        if (e.isComposing || e.keyCode === 229) {
          e.preventDefault();
          return;
        }

        // Timestamp-based guard: if compositionend just fired (within
        // 100ms), this Enter is from IME confirmation, not a real send.
        if (Date.now() - lastCompositionEndTime < 100) {
          e.preventDefault();
          return;
        }

        e.preventDefault();
        handler();
      });
    }

    /**
     * Bind event handlers
     * @param {Object} callbacks - Object containing callback functions
     */
    bindEvents(callbacks) {
      const {
        onClose,
        onClear,
        onSend,
        onInputChange,
        onThinkingToggle
      } = callbacks;

      // Bottom bar: typing/sending opens the panel and forwards the text
      const sendFromBar = () => {
        const text = this.barInput.value.trim();
        if (!text || this.isTyping) return;
        this.input.value = text;
        this.barInput.value = '';
        this._updateBarSendState();
        if (!this.isOpen) {
          this.openModal();
        }
        if (onSend) onSend();
      };

      this._attachEnterToSend(this.barInput, sendFromBar);
      this.barSendBtn.addEventListener('click', sendFromBar);
      this.barInput.addEventListener('input', () => this._updateBarSendState());
      this.barInput.addEventListener('focus', () => {
        if (!this.isOpen) this.openModal();
      });

      // Panel input: sends with context chips prepended
      const sendFromPanel = () => {
        const text = this.getInputValue().trim();
        if (!text || this.isTyping) return;
        if (this.contextChips.length > 0) {
          const contextBlock = this.contextChips
            .map((c) => '> ' + c.replace(/\n+/g, '\n> '))
            .join('\n');
          this.input.value = contextBlock + '\n\n' + text;
          this.clearChips();
        }
        if (onSend) onSend();
      };

      this._attachEnterToSend(this.input, sendFromPanel);
      this.sendBtn.addEventListener('click', sendFromPanel);

      // Auto-resize textarea
      this.input.addEventListener('input', () => this.autoResizeInput());
      if (onInputChange) {
        this.input.addEventListener('input', onInputChange);
      }

      // Close panel
      if (onClose) {
        this.closeBtn.addEventListener('click', (e) => {
          e.stopPropagation();
          onClose();
        });
      }

      // Clear conversation
      if (onClear) {
        this.clearBtn.addEventListener('click', (e) => {
          e.stopPropagation();
          onClear();
        });
      }

      // Toggle thinking mode
      if (onThinkingToggle) {
        this.thinkingBtn.addEventListener('click', (e) => {
          e.stopPropagation();
          onThinkingToggle();
        });
      }

      // Drag to resize panel
      this._initDragResize();

      // Text selection -> tooltip -> add context chip
      this._initSelectionTooltip();

      // Handle escape key
      document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && this.isOpen) {
          if (onClose) onClose();
        }
      });
    }

    /**
     * Update the bar send button active state
     */
    _updateBarSendState() {
      this.barSendBtn.classList.toggle('active', this.barInput.value.trim().length > 0);
    }

    /**
     * Initialize drag-to-resize for the side panel
     */
    _initDragResize() {
      const dragHandle = document.getElementById('askAiPanelDrag');
      if (!dragHandle) return;

      let dragging = false;
      let startX = 0;
      let startWidth = PANEL_DEFAULT_WIDTH;

      dragHandle.addEventListener('mousedown', (e) => {
        e.preventDefault();
        dragging = true;
        startX = e.clientX;
        startWidth = this.panel.offsetWidth;
        this.panel.classList.add('resizing');
        document.body.classList.add('ai-panel-resizing');
        document.body.style.cursor = 'col-resize';
        document.body.style.userSelect = 'none';
      });

      document.addEventListener('mousemove', (e) => {
        if (!dragging) return;
        const diff = startX - e.clientX;
        const maxWidth = Math.min(PANEL_MAX_WIDTH, window.innerWidth - 300);
        const newWidth = Math.min(Math.max(startWidth + diff, PANEL_MIN_WIDTH), maxWidth);
        document.body.style.setProperty('--ai-panel-width', newWidth + 'px');
      });

      document.addEventListener('mouseup', () => {
        if (!dragging) return;
        dragging = false;
        this.panel.classList.remove('resizing');
        document.body.classList.remove('ai-panel-resizing');
        document.body.style.cursor = '';
        document.body.style.userSelect = '';
        // Persist width
        const width = parseInt(getComputedStyle(document.body).getPropertyValue('--ai-panel-width'), 10);
        if (width && !Number.isNaN(width)) {
          try {
            localStorage.setItem(PANEL_WIDTH_KEY, String(width));
          } catch (err) {
            // Ignore storage errors (private mode, etc.)
          }
        }
      });
    }

    /**
     * Initialize the text selection tooltip
     */
    _initSelectionTooltip() {
      const tooltip = this.selectionTooltip;
      if (!tooltip) return;

      const handleSelection = () => {
        const selection = window.getSelection();
        const text = selection.toString().trim();

        if (text.length > 5 && text.length < 2000 && !this.root.contains(selection.anchorNode)) {
          const range = selection.getRangeAt(0);
          const rect = range.getBoundingClientRect();

          tooltip.style.top = (rect.top + window.scrollY - 40) + 'px';
          tooltip.style.left = (rect.left + window.scrollX + rect.width / 2) + 'px';
          tooltip.style.transform = 'translateX(-50%)';
          tooltip.classList.add('visible');

          tooltip.onclick = () => {
            tooltip.classList.remove('visible');
            selection.removeAllRanges();
            this.openModal();
            this.addChip(text);
            this.input.focus();
          };
        } else {
          tooltip.classList.remove('visible');
        }
      };

      document.addEventListener('mouseup', () => {
        setTimeout(handleSelection, 10);
      });

      document.addEventListener('mousedown', (e) => {
        if (!tooltip.contains(e.target)) {
          tooltip.classList.remove('visible');
        }
      });
    }

    /**
     * Add a context chip to the panel input area
     * @param {string} text - Context text
     */
    addChip(text) {
      this.contextChips.push(text);
      this.renderChips();
    }

    /**
     * Remove a context chip by index
     * @param {number} index - Chip index
     */
    removeChip(index) {
      this.contextChips.splice(index, 1);
      this.renderChips();
    }

    /**
     * Render context chips
     */
    renderChips() {
      if (!this.chipsRow) return;
      this.chipsRow.innerHTML = '';
      this.contextChips.forEach((text, i) => {
        const chip = document.createElement('span');
        chip.className = 'ai-panel-chip-item';
        const display = text.length > 30 ? text.substring(0, 30) + '...' : text;
        chip.innerHTML = `<span class="ai-panel-chip-item-text">${this.escapeHtml(display)}</span><span class="ai-panel-chip-item-close">&times;</span>`;
        chip.querySelector('.ai-panel-chip-item-close').addEventListener('click', () => {
          this.removeChip(i);
        });
        this.chipsRow.appendChild(chip);
      });
    }

    /**
     * Clear all context chips
     */
    clearChips() {
      this.contextChips = [];
      this.renderChips();
    }

    /**
     * Open the side panel
     */
    openModal() {
      this.isOpen = true;
      this.panel.classList.add('open');
      document.body.classList.add('ai-panel-open');
      this.scrollToBottom();
    }

    /**
     * Close the side panel
     */
    closeModal() {
      this.isOpen = false;
      this.panel.classList.remove('open');
      document.body.classList.remove('ai-panel-open');
    }

    /**
     * Toggle panel open/close
     */
    toggleModal() {
      if (this.isOpen) {
        this.closeModal();
      } else {
        this.openModal();
        this.input.focus();
      }
    }

    /**
     * Toggle expand/collapse state (wider panel)
     */
    toggleExpand() {
      this.isExpanded = !this.isExpanded;
      this.panel.classList.toggle('expanded', this.isExpanded);
    }

    /**
     * Auto-resize the input textarea
     */
    autoResizeInput() {
      this.input.style.height = 'auto';
      this.input.style.height = Math.min(this.input.scrollHeight, 100) + 'px';
    }

    /**
     * Add a message to the chat
     * @param {string} content - Message content
     * @param {string} type - Message type ('user' or 'assistant')
     * @param {string} messageId - Optional message ID
     * @returns {HTMLElement} The created message element
     */
    addMessage(content, type, messageId = null) {
      const messageDiv = document.createElement('div');
      messageDiv.className = `ask-ai-message ${type}`;

      // Generate unique message ID if not provided
      const msgId = messageId || `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      messageDiv.setAttribute('data-message-id', msgId);

      // For assistant messages, render as markdown; for user messages, keep as plain text
      if (type === 'assistant') {
        messageDiv.innerHTML = this.renderMarkdown(content);
      } else {
        messageDiv.textContent = content;
      }

      // Add to DOM first
      this.messagesContainer.appendChild(messageDiv);

      // Then add feedback buttons for assistant messages after DOM insertion
      if (type === 'assistant') {
        // Use setTimeout to ensure DOM is fully updated
        setTimeout(() => {
          this.addFeedbackButtons(messageDiv, msgId, content);
        }, 0);
      }

      this.scrollToBottom();

      // Store message
      this.messages.push({ content, type, timestamp: Date.now(), messageId: msgId });

      return messageDiv;
    }

    /**
     * Add feedback buttons to assistant message
     * @param {HTMLElement} messageDiv - Message element
     * @param {string} messageId - Message ID
     * @param {string} content - Message content for copying
     */
    addFeedbackButtons(messageDiv, messageId, content) {
      if (!messageDiv || !messageDiv.parentNode) {
        console.warn('Message div not in DOM yet, retrying...');
        // Retry after a short delay
        setTimeout(() => {
          if (messageDiv && messageDiv.parentNode) {
            this.addFeedbackButtons(messageDiv, messageId, content);
          }
        }, 10);
        return;
      }

      // Check if wrapper already exists to avoid duplicate creation
      let messageWrapper = messageDiv.parentNode;
      if (!messageWrapper || !messageWrapper.classList.contains('ask-ai-message-wrapper')) {
        // Create wrapper
        messageWrapper = document.createElement('div');
        messageWrapper.className = 'ask-ai-message-wrapper';

        // Insert wrapper and move message
        const parentContainer = messageDiv.parentNode;
        parentContainer.insertBefore(messageWrapper, messageDiv);
        messageWrapper.appendChild(messageDiv);
      }

      // Check if feedback buttons already exist to avoid duplicate addition
      let feedbackDiv = messageWrapper.querySelector('.ask-ai-feedback-actions');
      if (!feedbackDiv) {
        feedbackDiv = document.createElement('div');
        feedbackDiv.className = 'ask-ai-feedback-actions';
        feedbackDiv.innerHTML = `
        <button class="ask-ai-feedback-btn like" data-feedback="like" title="${this.i18n.like}">
          ${LIKE_ICON}
        </button>
        <button class="ask-ai-feedback-btn dislike" data-feedback="dislike" title="${this.i18n.dislike}">
          ${DISLIKE_ICON}
        </button>
        <button class="ask-ai-feedback-btn copy" title="${this.i18n.copyMarkdown}">
          ${COPY_ICON}
        </button>
      `;

        // Append to wrapper
        messageWrapper.appendChild(feedbackDiv);
      }

      // Store content for copying
      feedbackDiv.setAttribute('data-content', content);
    }


    /**
     * Update message content while preserving tool calls and feedback buttons
     * Content is organized in segments: each tool call creates a new segment
     * @param {HTMLElement} messageDiv - Message element
     * @param {string} content - New content (cumulative from stream)
     * @param {boolean} addSuffix - Whether to add helpSuffix (default: false, used during streaming)
     */
    updateMessageContent(messageDiv, content, addSuffix = false) {
      if (!messageDiv) return;

      // Remove typing indicator if present
      const typingIndicator = messageDiv.querySelector('.typing-indicator');
      if (typingIndicator) {
        typingIndicator.remove();
      }
      
      // Only append helpSuffix when explicitly requested (at the end of response)
      const contentToRender = addSuffix ? content + (this.i18n.helpSuffix || '') : content;
      
      // Find the last "block" element (tool container or thinking panel)
      // Content should always be placed after the last block element
      const toolContainers = messageDiv.querySelectorAll('.tool-calls-inline');
      const thinkingContainers = messageDiv.querySelectorAll('.thinking-inline');
      const lastToolContainer = toolContainers.length > 0 ? toolContainers[toolContainers.length - 1] : null;
      const lastThinkingContainer = thinkingContainers.length > 0 ? thinkingContainers[thinkingContainers.length - 1] : null;
      
      // Determine the last block element by comparing DOM positions
      let lastBlockElement = null;
      if (lastToolContainer && lastThinkingContainer) {
        // Both exist - find which one comes last in DOM order
        const position = lastToolContainer.compareDocumentPosition(lastThinkingContainer);
        lastBlockElement = (position & Node.DOCUMENT_POSITION_FOLLOWING) ? lastThinkingContainer : lastToolContainer;
      } else {
        lastBlockElement = lastThinkingContainer || lastToolContainer;
      }
      
      if (lastBlockElement) {
        // Find or create content wrapper AFTER the last block element
        let contentWrapper = lastBlockElement.nextElementSibling;
        if (!contentWrapper || !contentWrapper.classList.contains('message-content-segment')) {
          contentWrapper = document.createElement('div');
          contentWrapper.className = 'message-content-segment';
          lastBlockElement.after(contentWrapper);
        }
        
        // Calculate what content belongs to this segment
        const segmentContent = this.extractContentAfterTools(messageDiv, content);
        contentWrapper.innerHTML = this.renderMarkdown(segmentContent);
      } else {
        // No block elements - find or create the first content segment
        let contentWrapper = messageDiv.querySelector('.message-content-segment');
        if (!contentWrapper) {
          contentWrapper = document.createElement('div');
          contentWrapper.className = 'message-content-segment';
          messageDiv.appendChild(contentWrapper);
        }
        contentWrapper.innerHTML = this.renderMarkdown(contentToRender);
      }
      
      // Store full content for copying
      messageDiv.setAttribute('data-full-content', content);
      
      this.scrollToBottom();
    }

    /**
     * Extract content that should appear after the last tool call
     * This handles the cumulative content from streaming
     * @param {HTMLElement} messageDiv - Message element
     * @param {string} fullContent - Full cumulative content
     * @returns {string} Content for the current segment
     */
    extractContentAfterTools(messageDiv, fullContent) {
      // Get the content length that was rendered before the last tool call
      const lastRenderedLength = parseInt(messageDiv.getAttribute('data-content-before-last-tool') || '0', 10);
      
      // Return only the new content after the last tool call
      if (lastRenderedLength > 0 && lastRenderedLength < fullContent.length) {
        return fullContent.substring(lastRenderedLength);
      }
      
      // If no previous content recorded, return full content
      return fullContent;
    }

    /**
     * Finalize message content by adding helpSuffix
     * Called when response is complete - just adds helpSuffix to the last content segment
     * The content segments are already correctly rendered during streaming
     * @param {HTMLElement} messageDiv - Message element
     * @param {string} content - Final content (may be just the last segment from server)
     */
    finalizeMessage(messageDiv, content) {
      if (!messageDiv) return;
      
      const messageId = messageDiv.getAttribute('data-message-id');
      
      // Find the last block element (tool container or thinking panel)
      const toolContainers = messageDiv.querySelectorAll('.tool-calls-inline');
      const thinkingContainers = messageDiv.querySelectorAll('.thinking-inline');
      const lastToolContainer = toolContainers.length > 0 ? toolContainers[toolContainers.length - 1] : null;
      const lastThinkingContainer = thinkingContainers.length > 0 ? thinkingContainers[thinkingContainers.length - 1] : null;
      
      let lastBlockElement = null;
      if (lastToolContainer && lastThinkingContainer) {
        const position = lastToolContainer.compareDocumentPosition(lastThinkingContainer);
        lastBlockElement = (position & Node.DOCUMENT_POSITION_FOLLOWING) ? lastThinkingContainer : lastToolContainer;
      } else {
        lastBlockElement = lastThinkingContainer || lastToolContainer;
      }
      
      if (!lastBlockElement) {
        // No block elements - simple case, just render all content with suffix
        let contentWrapper = messageDiv.querySelector('.message-content-segment');
        if (!contentWrapper) {
          contentWrapper = document.createElement('div');
          contentWrapper.className = 'message-content-segment';
          messageDiv.appendChild(contentWrapper);
        }
        contentWrapper.innerHTML = this.renderMarkdown(content + (this.i18n.helpSuffix || ''));
      } else {
        // Has block elements - find the last content segment after the last block
        let lastContentSegment = lastBlockElement.nextElementSibling;
        
        if (lastContentSegment && lastContentSegment.classList.contains('message-content-segment')) {
          const fullContent = messageDiv.getAttribute('data-full-content') || content;
          const contentBeforeLastTool = parseInt(messageDiv.getAttribute('data-content-before-last-tool') || '0', 10);
          const segmentContent = contentBeforeLastTool > 0 && contentBeforeLastTool < fullContent.length 
            ? fullContent.substring(contentBeforeLastTool) 
            : fullContent;
          lastContentSegment.innerHTML = this.renderMarkdown(segmentContent + (this.i18n.helpSuffix || ''));
        } else {
          // No content segment after last block - check if there should be one
          const fullContent = messageDiv.getAttribute('data-full-content') || content;
          const contentBeforeLastTool = parseInt(messageDiv.getAttribute('data-content-before-last-tool') || '0', 10);
          const segmentContent = contentBeforeLastTool > 0 && contentBeforeLastTool < fullContent.length 
            ? fullContent.substring(contentBeforeLastTool) 
            : '';
          
          if (segmentContent.trim()) {
            lastContentSegment = document.createElement('div');
            lastContentSegment.className = 'message-content-segment';
            lastContentSegment.innerHTML = this.renderMarkdown(segmentContent + (this.i18n.helpSuffix || ''));
            lastBlockElement.after(lastContentSegment);
          }
        }
      }
      
      // Store full content for copying (use existing if available)
      const existingFullContent = messageDiv.getAttribute('data-full-content');
      if (!existingFullContent) {
        messageDiv.setAttribute('data-full-content', content);
      }
      
      // Add feedback buttons
      if (messageId) {
        const fullContent = messageDiv.getAttribute('data-full-content') || content;
        this.addFeedbackButtons(messageDiv, messageId, fullContent);
      }
      
      this.scrollToBottom();
    }

    /**
     * Show typing indicator
     * @returns {HTMLElement} The typing indicator element
     */
    showTypingIndicator() {
      this.isTyping = true;
      this.sendBtn.disabled = true;

      const typingDiv = document.createElement('div');
      typingDiv.className = 'ask-ai-message assistant typing';
      typingDiv.id = 'typingIndicator';
      typingDiv.innerHTML = `
      <div class="typing-indicator">
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
      </div>
    `;

      this.messagesContainer.appendChild(typingDiv);
      this.scrollToBottom();

      return typingDiv;
    }

    /**
     * Hide typing indicator
     */
    hideTypingIndicator() {
      this.isTyping = false;
      this.sendBtn.disabled = false;

      const typingIndicator = document.getElementById('typingIndicator');
      if (typingIndicator) {
        typingIndicator.remove();
      }
    }

    /**
     * Toggle thinking mode on/off
     */
    toggleThinking() {
      this.enableThinking = !this.enableThinking;
      if (this.enableThinking) {
        this.thinkingBtn.classList.add('active');
      } else {
        this.thinkingBtn.classList.remove('active');
      }
    }

    /**
     * Create a new thinking container inside message bubble as a collapsible panel.
     * Each reasoning phase gets its own container.
     * @param {HTMLElement} messageDiv - Message element to add thinking info to
     * @returns {HTMLElement} The created thinking container
     */
    createThinkingContainer(messageDiv) {
      if (!messageDiv) return null;

      // Record current content length before adding thinking block
      const currentFullContent = messageDiv.getAttribute('data-full-content') || '';
      messageDiv.setAttribute('data-content-before-last-tool', currentFullContent.length.toString());

      const thinkingContainer = document.createElement('div');
      thinkingContainer.className = 'thinking-inline';

      // Add collapsible header
      const header = document.createElement('div');
      header.className = 'thinking-inline-header';
      header.innerHTML = `
      <span class="thinking-inline-title">💭 ${this.i18n.thinkingContent}</span>
      <button class="thinking-inline-toggle">▼</button>
    `;
      thinkingContainer.appendChild(header);

      // Add content container
      const thinkingContentDiv = document.createElement('div');
      thinkingContentDiv.className = 'thinking-inline-content';
      thinkingContainer.appendChild(thinkingContentDiv);

      // Append thinking container at the end of messageDiv
      messageDiv.appendChild(thinkingContainer);

      // Add toggle functionality
      const toggleBtn = header.querySelector('.thinking-inline-toggle');
      toggleBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        const contentDiv = thinkingContainer.querySelector('.thinking-inline-content');
        const isCollapsed = contentDiv.style.display === 'none';
        contentDiv.style.display = isCollapsed ? 'block' : 'none';
        toggleBtn.textContent = isCollapsed ? '▼' : '▶';
        thinkingContainer.classList.toggle('collapsed', !isCollapsed);
      });

      return thinkingContainer;
    }

    /**
     * Append delta text to an existing thinking container
     * @param {string} thinkingText - Incremental thinking text (delta)
     * @param {HTMLElement} thinkingContainer - The active thinking container
     */
    appendThinkingContent(thinkingText, thinkingContainer) {
      if (!thinkingContainer) return;

      const thinkingContentDiv = thinkingContainer.querySelector('.thinking-inline-content');
      if (!thinkingContentDiv) return;

      const currentText = thinkingContentDiv.getAttribute('data-raw-text') || '';
      const updatedText = currentText + thinkingText;
      thinkingContentDiv.setAttribute('data-raw-text', updatedText);
      thinkingContentDiv.innerHTML = this.renderMarkdown(updatedText);

      this.scrollToBottom();
    }

    /**
     * Finalize a specific thinking container - collapse it when done
     * @param {HTMLElement} thinkingContainer - The thinking container to finalize
     */
    finalizeThinking(thinkingContainer) {
      if (!thinkingContainer) return;

      const contentDiv = thinkingContainer.querySelector('.thinking-inline-content');
      const toggleBtn = thinkingContainer.querySelector('.thinking-inline-toggle');
      if (contentDiv && toggleBtn) {
        contentDiv.style.display = 'none';
        toggleBtn.textContent = '▶';
        thinkingContainer.classList.add('collapsed');
      }
    }

    /**
     * Add tool call info inside message bubble
     * Consecutive tool calls go into the same tool container
     * A new tool container is only created when there's text content between tool calls
     * @param {string} toolName - Name of the tool being used
     * @param {Object} toolArgs - Tool arguments
     * @param {HTMLElement} messageDiv - Message element to add tool info to
     */
    addToolCall(toolName, toolArgs, messageDiv) {
      if (!messageDiv) return;

      // Record current content length before adding tool call
      // This is used by updateMessageContent to know where to split content
      const currentFullContent = messageDiv.getAttribute('data-full-content') || '';
      messageDiv.setAttribute('data-content-before-last-tool', currentFullContent.length.toString());

      // Check if we should reuse the last tool container or create a new one
      // Reuse if: the last child is a tool container (no text content in between)
      let toolContainer = null;
      const lastChild = messageDiv.lastElementChild;
      
      if (lastChild && lastChild.classList.contains('tool-calls-inline')) {
        // Reuse existing tool container (consecutive tool calls)
        toolContainer = lastChild;
      } else {
        // Create a new tool container (first tool call or there's text content before this)
        toolContainer = document.createElement('div');
        toolContainer.className = 'tool-calls-inline';
        
        // Add collapsible header
        const header = document.createElement('div');
        header.className = 'tool-calls-inline-header';
        header.innerHTML = `
        <span class="tool-calls-inline-title">🔧 ${this.i18n.toolCalls}</span>
        <button class="tool-calls-inline-toggle">▼</button>
      `;
        toolContainer.appendChild(header);
        
        // Add content container
        const toolContentDiv = document.createElement('div');
        toolContentDiv.className = 'tool-calls-inline-content';
        toolContainer.appendChild(toolContentDiv);
        
        // Append tool container at the end of messageDiv
        messageDiv.appendChild(toolContainer);
        
        // Add toggle functionality
        const toggleBtn = header.querySelector('.tool-calls-inline-toggle');
        toggleBtn.addEventListener('click', (e) => {
          e.stopPropagation();
          const contentDiv = toolContainer.querySelector('.tool-calls-inline-content');
          const isCollapsed = contentDiv.style.display === 'none';
          contentDiv.style.display = isCollapsed ? 'block' : 'none';
          toggleBtn.textContent = isCollapsed ? '▼' : '▶';
          toolContainer.classList.toggle('collapsed', !isCollapsed);
        });
      }
      
      // Get the content container from the tool container
      const toolContent = toolContainer.querySelector('.tool-calls-inline-content');

      // Create tool call item
      const toolId = `tool_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      const toolItem = document.createElement('div');
      toolItem.className = 'tool-call-inline running';
      toolItem.setAttribute('data-tool-id', toolId);
      
      // Format arguments for display (compact)
      let argsPreview = '';
      if (toolArgs && Object.keys(toolArgs).length > 0) {
        const argsList = Object.entries(toolArgs).map(([key, value]) => {
          const displayValue = typeof value === 'string' && value.length > 50 
            ? value.substring(0, 50) + '...' 
            : JSON.stringify(value);
          return `${key}: ${this.escapeHtml(displayValue)}`;
        }).join(', ');
        argsPreview = `<div class="tool-args-preview">${argsList}</div>`;
      }

      toolItem.innerHTML = `
      <div class="tool-inline-header">
        <span class="tool-name-inline">${this.escapeHtml(toolName)}</span>
        <span class="tool-status-inline running"></span>
      </div>
      ${argsPreview}
    `;

      toolContent.appendChild(toolItem);
      this.scrollToBottom();

      return toolId;
    }

    /**
     * Mark a tool call as completed
     * @param {string} toolId - Tool ID to mark as done
     */
    markToolCallDone(toolId) {
      const toolItem = this.messagesContainer.querySelector(`[data-tool-id="${toolId}"]`);
      if (toolItem) {
        toolItem.classList.remove('running');
        const statusSpan = toolItem.querySelector('.tool-status-inline');
        if (statusSpan) {
          statusSpan.textContent = this.i18n.done;
          statusSpan.classList.remove('running');
        }
      }
    }

    /**
     * Collapse all fully-completed tool containers in a message
     * Called when a non-tool-call phase starts (text content or thinking)
     * @param {HTMLElement} messageDiv - Message element
     */
    collapseCompletedToolContainers(messageDiv) {
      if (!messageDiv) return;

      const toolContainers = messageDiv.querySelectorAll('.tool-calls-inline');
      toolContainers.forEach(toolContainer => {
        // Skip already collapsed containers
        if (toolContainer.classList.contains('collapsed')) return;

        // Check if all tools in this container are done (no running ones)
        const remainingRunning = toolContainer.querySelectorAll('.tool-call-inline.running');
        if (remainingRunning.length === 0) {
          const contentDiv = toolContainer.querySelector('.tool-calls-inline-content');
          const toggleBtn = toolContainer.querySelector('.tool-calls-inline-toggle');
          if (contentDiv && toggleBtn) {
            contentDiv.style.display = 'none';
            toggleBtn.textContent = '▶';
            toolContainer.classList.add('collapsed');
          }
        }
      });
    }

    /**
     * Scroll messages container to bottom
     */
    scrollToBottom() {
      setTimeout(() => {
        this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
      }, 100);
    }

    /**
     * Clear all messages from UI
     */
    clearMessages() {
      this.messages = [];
      const existingMessages = this.messagesContainer.querySelectorAll('.ask-ai-message, .ask-ai-message-wrapper');
      existingMessages.forEach(msg => msg.remove());
      this.clearChips();
      // Note: Welcome message is intentionally kept - it will be updated by addWelcomeMessage()
    }

    /**
     * Add welcome message
     * @param {boolean} apiConnected - Whether API is connected
     */
    addWelcomeMessage(apiConnected) {
      // Always show welcome message, regardless of history
      let welcomeElement = this.messagesContainer.querySelector('.ask-ai-welcome');
      
      // If welcome element doesn't exist, create it
      if (!welcomeElement) {
        welcomeElement = document.createElement('div');
        welcomeElement.className = 'ask-ai-welcome';
        // Insert at the beginning of messages container
        this.messagesContainer.insertBefore(welcomeElement, this.messagesContainer.firstChild);
      }
      
      // Update welcome message content based on connection status
      if (apiConnected) {
        welcomeElement.innerHTML = this.i18n.welcomeConnected;
      } else {
        welcomeElement.innerHTML = this.i18n.welcomeOffline;
      }
    }

    /**
     * Render markdown text to HTML
     * @param {string} text - Markdown text
     * @returns {string} HTML string
     */
    renderMarkdown(text) {
      if (!text) return '';

      try {
        const renderer = new marked.Renderer();

        // Custom heading renderer - use CSS classes instead of inline styles
        renderer.heading = (token) => {
          const escapedText = this.escapeHtml(token.text);
          return `<h${token.depth}>${escapedText}</h${token.depth}>`;
        };

        // Custom link renderer - open in new tab
        renderer.link = (token) => {
          const href = token.href;
          const title = token.title ? ` title="${this.escapeHtml(token.title)}"` : '';
          const text = token.text;
          return `<a href="${href}"${title} target="_blank" rel="noopener noreferrer">${text}</a>`;
        };

        return marked.parse(text, { renderer });
      } catch (error) {
        console.error('Markdown rendering error:', error);
        return this.escapeHtml(text).replace(/\n/g, '<br>');
      }
    }

    /**
     * Escape HTML special characters
     * @param {string} text - Text to escape
     * @returns {string} Escaped text
     */
    escapeHtml(text) {
      const div = document.createElement('div');
      div.textContent = text;
      return div.innerHTML;
    }

    /**
     * Observe theme changes from the Sphinx theme.
     * The widget adapts through CSS variables bound to
     * html[data-theme], so no class juggling is required; this method is
     * kept for API compatibility.
     */
    observeThemeChanges() {
      // No-op: styles follow html[data-theme] via CSS variables.
    }

    /**
     * Get current input value
     * @returns {string} Input value
     */
    getInputValue() {
      return this.input.value;
    }

    /**
     * Clear input value
     */
    clearInput() {
      this.input.value = '';
      this.autoResizeInput();
    }
  }

  /**
   * Ask AI Widget - Main Controller
   * 
   * This is the main entry point that integrates all modules:
   * - I18N: Internationalization
   * - API: Communication with backend
   * - UI: User interface management
   */


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

  return AskAIWidget;

})();
