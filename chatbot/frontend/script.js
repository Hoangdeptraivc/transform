let paragraph = '';

document.addEventListener('DOMContentLoaded', function() {
    // Các biến
    const chatToggle = document.getElementById('chatToggle');
    const chatContainer = document.querySelector('.chat-container');
    const chatHeader = document.getElementById('chatHeader');
    const chatBody = document.getElementById('chatBody');
    const minimizeBtn = document.getElementById('minimizeChat');
    const closeBtn = document.getElementById('closeChat');
    const userInput = document.getElementById('userInput');
    const sendBtn = document.getElementById('sendMessage');
    const chatMessages = document.getElementById('chatMessages');
    
    // Trạng thái ban đầu
    let isChatVisible = true;
    let isMinimized = false;
    
    // Hiển thị/ẩn chat
    chatToggle.addEventListener('click', function() {
        isChatVisible = !isChatVisible;
        if (isChatVisible) {
            chatContainer.classList.add('chat-visible');
            chatContainer.classList.remove('chat-hidden');
        } else {
            chatContainer.classList.add('chat-hidden');
            chatContainer.classList.remove('chat-visible');
        }
    });
    
    // Thu nhỏ/phóng to chat
    minimizeBtn.addEventListener('click', function() {
        isMinimized = !isMinimized;
        if (isMinimized) {
            chatContainer.classList.add('chat-minimized');
        } else {
            chatContainer.classList.remove('chat-minimized');
        }
    });
    
    // Đóng chat
    closeBtn.addEventListener('click', function() {
        chatContainer.classList.add('chat-hidden');
        chatToggle.style.display = 'block';
        isChatVisible = false;
    });
    
    // Mở chat từ header
    chatHeader.addEventListener('click', function() {
        if (isMinimized) {
            isMinimized = false;
            chatContainer.classList.remove('chat-minimized');
        }
    });
    
    // Gửi tin nhắn
    async function sendMessage() {
    const message = userInput.value.trim();
    if (message !== '') {
        // Hiển thị tin nhắn người dùng
        addMessage(message, 'user');
        userInput.value = '';

        try {
            // Gửi tin nhắn đến backend
            paragraph += message;
            const response = await fetch('http://127.0.0.1:5000/chatbot', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: paragraph })
            });

            const data = await response.json();

            // Hiển thị phản hồi từ chatbot
            if (data.reply) {
                addMessage(data.reply, 'bot');
            } else {
                addMessage("Xin lỗi, tôi không hiểu câu hỏi của bạn.", 'bot');
            }

        } catch (error) {
            console.error('Lỗi khi gửi tin nhắn:', error);
            addMessage("Xin lỗi, có lỗi xảy ra khi kết nối với chatbot.", 'bot');
        }
    }
}
    
    // Gửi tin nhắn khi nhấn nút hoặc Enter
    sendBtn.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });
    
    // Thêm tin nhắn vào khung chat
    function addMessage(text, sender) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', sender + '-message');
        
        const messagePara = document.createElement('p');
        messagePara.textContent = text;
        
        messageDiv.appendChild(messagePara);
        chatMessages.appendChild(messageDiv);
        
        // Cuộn xuống dưới cùng
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    // Hàm mô phỏng phản hồi từ chatbot (thay thế bằng API thực tế)
    function getBotResponse(userMessage) {
        const lowerMessage = userMessage.toLowerCase();
        
        if (lowerMessage.includes('xin chào') || lowerMessage.includes('hello')) {
            return "Xin chào! Tôi có thể giúp gì cho bạn?";
        } else if (lowerMessage.includes('cảm ơn')) {
            return "Không có gì! Tôi rất vui khi được giúp đỡ bạn.";
        } else if (lowerMessage.includes('tạm biệt')) {
            return "Tạm biệt! Nếu bạn cần thêm trợ giúp, cứ quay lại nhé!";
        } else if (lowerMessage.includes('giờ') || lowerMessage.includes('thời gian')) {
            return "Bây giờ là " + new Date().toLocaleTimeString() + ".";
        } else if (lowerMessage.includes('ngày') || lowerMessage.includes('hôm nay')) {
            return "Hôm nay là ngày " + new Date().toLocaleDateString() + ".";
        } else {
            return "Tôi không chắc mình hiểu câu hỏi của bạn. Bạn có thể diễn đạt lại không?";
        }
    }
});