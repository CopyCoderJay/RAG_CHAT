// Chat functionality JavaScript
let currentChatId = null;

// Chat item click handler
document.addEventListener('DOMContentLoaded', function() {
    const chatItems = document.querySelectorAll('.chat-item');
    chatItems.forEach(item => {
        item.addEventListener('click', function() {
            const chatId = this.dataset.chatId;
            loadChat(chatId);
        });
    });
});

function createNewChat() {
    fetch('/api/chat/create/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': getCookie('csrftoken')
        },
        body: JSON.stringify({
            title: 'New Chat'
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert('Error creating chat: ' + data.error);
        } else {
            loadChat(data.id);
            location.reload(); // Refresh to show new chat in sidebar
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('Error creating chat');
    });
}

function loadChat(chatId) {
    currentChatId = chatId;
    document.getElementById('chat-title').textContent = 'Chat ' + chatId.substring(0, 8);
    document.getElementById('message-input').disabled = false;
    document.getElementById('send-btn').disabled = false;
    
    // Load messages
    fetch(`/api/chat/${chatId}/messages/`)
    .then(response => response.json())
    .then(messages => {
        const chatMessages = document.getElementById('chat-messages');
        chatMessages.innerHTML = '';
        if (messages.length === 0) {
            chatMessages.innerHTML = '<div class="text-center text-muted"><i class="fas fa-comment-dots fa-2x mb-2"></i><p>No messages yet. Start the conversation!</p></div>';
        } else {
            messages.forEach(message => {
                addMessageToChat(message.role, message.content, message.sources);
            });
        }
    })
    .catch(error => {
        console.error('Error loading messages:', error);
    });
}

function sendMessage() {
    const input = document.getElementById('message-input');
    const message = input.value.trim();
    if (!message || !currentChatId) return;
    
    // Add user message to chat
    addMessageToChat('user', message);
    input.value = '';
    
    // Show loading indicator
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'message assistant';
    loadingDiv.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Thinking...';
    document.getElementById('chat-messages').appendChild(loadingDiv);
    
    // Send to backend
    fetch('/api/chat/send-message/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': getCookie('csrftoken')
        },
        body: JSON.stringify({
            chat_id: currentChatId,
            message: message,
            use_rag: document.getElementById('use-rag').checked
        })
    })
    .then(response => response.json())
    .then(data => {
        // Remove loading indicator
        const loadingDiv = document.querySelector('.message.assistant:last-child');
        if (loadingDiv && loadingDiv.innerHTML.includes('spinner')) {
            loadingDiv.remove();
        }
        
        if (data.error) {
            addMessageToChat('assistant', 'Error: ' + data.error);
        } else {
            addMessageToChat('assistant', data.response, data.sources);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        addMessageToChat('assistant', 'Error: Failed to get response');
    });
}

function addMessageToChat(role, content, sources = []) {
    const chatMessages = document.getElementById('chat-messages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;
    
    let sourcesHtml = '';
    if (sources && sources.length > 0) {
        sourcesHtml = '<div class="sources"><small><i class="fas fa-book me-1"></i>Sources: ';
        sources.forEach((source, index) => {
            sourcesHtml += `[S${index + 1}] Page ${source.page} `;
        });
        sourcesHtml += '</small></div>';
    }
    
    messageDiv.innerHTML = `
        <div class="message-content">
            <strong><i class="fas fa-${role === 'user' ? 'user' : 'robot'} me-1"></i>${role.charAt(0).toUpperCase() + role.slice(1)}:</strong>
            <p class="mb-1">${content}</p>
            ${sourcesHtml}
        </div>
    `;
    
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Enter key handler
document.getElementById('message-input').addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
        sendMessage();
    }
});

function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}

