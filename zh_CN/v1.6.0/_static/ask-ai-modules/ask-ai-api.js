/**
 * Ask AI Widget - API Communication Layer
 */

export class AskAIApi {
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
