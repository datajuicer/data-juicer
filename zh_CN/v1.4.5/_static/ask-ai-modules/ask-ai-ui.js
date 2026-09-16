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

export class AskAIUI {
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
