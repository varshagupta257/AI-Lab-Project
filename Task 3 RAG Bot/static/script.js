// ======================================
// Enterprise AI - Frontend
// ======================================

const sendBtn = document.getElementById("sendBtn");
const userInput = document.getElementById("userInput");
const messages = document.getElementById("messages");

const newChatBtn = document.getElementById("newChat");
const historyDiv = document.getElementById("history");

const uploadBtn = document.getElementById("uploadBtn");
const fileUpload = document.getElementById("fileUpload");

const knowledgeBase = document.getElementById("knowledgeBase");

let chats = [];
let currentChat = null;


// ======================================
// Create New Chat
// ======================================

function createNewChat() {

    currentChat = {
        id: Date.now(),
        title: "New Chat",
        messages: []
    };

    chats.unshift(currentChat);

    renderHistory();

    renderMessages();
}


// ======================================
// Render Sidebar History
// ======================================

function renderHistory() {

    historyDiv.innerHTML = "";

    chats.forEach(chat => {

        const item = document.createElement("div");

        item.className = "history-item";

        item.innerText = chat.title;

        item.onclick = () => {

            currentChat = chat;

            renderMessages();

        };

        historyDiv.appendChild(item);

    });

}


// ======================================
// Render Chat Messages
// ======================================

function renderMessages() {

    messages.innerHTML = "";

    currentChat.messages.forEach(msg => {

        const div = document.createElement("div");

        div.className =
            msg.role === "user"
                ? "user-message"
                : "bot-message";

        div.innerText = msg.content;

        messages.appendChild(div);

    });

    messages.scrollTop = messages.scrollHeight;

}


// ======================================
// Add User Message
// ======================================

function addUserMessage(text) {

    currentChat.messages.push({

        role: "user",

        content: text

    });

    renderMessages();

}


// ======================================
// Add Bot Message
// ======================================

function addBotMessage(text) {

    currentChat.messages.push({

        role: "assistant",

        content: text

    });

    renderMessages();

}


// ======================================
// Send Message
// ======================================

async function sendMessage() {

    const text = userInput.value.trim();

    if (text === "")
        return;

    if (currentChat.messages.length === 0) {

        currentChat.title = text.substring(0, 30);

        renderHistory();

    }

    addUserMessage(text);

    userInput.value = "";

    addBotMessage("Thinking...");

    const loadingIndex = currentChat.messages.length - 1;

    try {

        const response = await fetch("/chat", {

            method: "POST",

            headers: {

                "Content-Type": "application/json"

            },

            body: JSON.stringify({

                message: text

            })

        });

        const data = await response.json();

        currentChat.messages[loadingIndex].content =
            data.response;

        renderMessages();

    }
    catch (err) {

        currentChat.messages[loadingIndex].content =
            "Unable to communicate with AI.";

        renderMessages();

        console.error(err);

    }

}


// ======================================
// Upload Document
// ======================================

if (uploadBtn && fileUpload) {

    uploadBtn.onclick = () => {

        fileUpload.click();

    };

    fileUpload.addEventListener("change", async () => {

        if (fileUpload.files.length === 0)
            return;

        const form = new FormData();

        form.append("file", fileUpload.files[0]);

        uploadBtn.innerText = "Uploading...";

        try {

            const response = await fetch("/upload", {

                method: "POST",

                body: form

            });

            const result = await response.json();

            alert(result.message);

            loadKnowledgeBase();

        }
        catch (err) {

            alert("Upload Failed.");

            console.error(err);

        }

        uploadBtn.innerText = "📁 Upload Document";

    });

}


// ======================================
// Knowledge Base
// ======================================

async function loadKnowledgeBase() {

    if (!knowledgeBase)
        return;

    knowledgeBase.innerHTML = "";

    try {

        const response = await fetch("/documents");

        const data = await response.json();

        if (data.documents.length === 0) {

            knowledgeBase.innerHTML =
                "<div class='empty'>No documents uploaded.</div>";

            return;

        }

        data.documents.forEach(file => {

            const div = document.createElement("div");

            div.className = "doc-item";

            div.innerHTML = "📄 " + file;

            knowledgeBase.appendChild(div);

        });

    }
    catch {

        knowledgeBase.innerHTML =
            "<div class='empty'>Unable to load documents.</div>";

    }

}


// ======================================
// Events
// ======================================

sendBtn.onclick = sendMessage;

userInput.addEventListener("keypress", function (event) {

    if (event.key === "Enter") {

        sendMessage();

    }

});

newChatBtn.onclick = createNewChat;


// ======================================
// Startup
// ======================================

createNewChat();

loadKnowledgeBase();