import React, { useState, useEffect, useRef } from 'react';
import { Link } from 'react-router-dom';
import './Chatbot.css';

const BotMessageContent = ({ message }) => {
  if (message.content) {
    return <div dangerouslySetInnerHTML={{ __html: message.content }} />;
  }

  if (message.data) {
    if (message.data.type === 'title_recommendation') {
      return (
        <div>
          <p className="recommendation-title">{message.data.response_title}</p>
          <div className="recommendation-list">
            {message.data.recommendations.map((book, index) => (
              <div key={index} className="recommendation-card">
                <span className="serial-number">{index + 1}.</span>
                <a
                  href="#"
                  className="book-detail-link"
                  data-title={book.title}
                >
                  {book.title ? book.title.toUpperCase() : 'UNKNOWN'}
                </a>
              </div>
            ))}
          </div>
        </div>
      );
    }

    if (message.data.type === 'gemini_details') {
      const { title, author, category, description } = message.data.details || {};
      return (
        <div className="details-card">
          <p className="details-title">{title ? title.toUpperCase() : 'UNKNOWN'}</p>
          <p><strong>Author:</strong> {author || 'N/A'}</p>
          <p><strong>Category:</strong> {category || 'N/A'}</p>
          <p className="details-description">{description || 'No description available.'}</p>
        </div>
      );
    }
  }

  return <div>...</div>;
};


const Main = () => {
  const [messages, setMessages] = useState([]);
  const [userInput, setUserInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const chatWindowRef = useRef(null);

  const apiUrl = process.env.REACT_APP_API_URL || 'http://127.0.0.1:5001';

  useEffect(() => {
    setMessages([{ author: 'bot', content: "Welcome to BookVoyager.AI! Enter a book title to get recommendations." }]);
  }, []);

  useEffect(() => {
    if (chatWindowRef.current) {
      chatWindowRef.current.scrollTop = chatWindowRef.current.scrollHeight;
    }
  }, [messages]);

  const fetchBookDetails = async (title) => {
    setIsLoading(true);
    setMessages(prev => [...prev, { author: 'user', content: `Tell me more about "${title.toUpperCase()}"` }]);
    try {
        const response = await fetch(`${apiUrl}/get_book_details`, { 
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ book_title: title }),
        });
        const result = await response.json();
        const botMessage = result.error
            ? { author: 'bot', content: result.error }
            : { author: 'bot', data: { type: 'gemini_details', details: result } };
        setMessages(prev => [...prev, botMessage]);
    } catch (error) {
        setMessages(prev => [...prev, { author: 'bot', content: "Sorry, I couldn't fetch the details." }]);
    } finally {
        setIsLoading(false);
    }
  };
  
  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!userInput.trim()) return;
    
    setMessages(prev => [...prev, { author: 'user', content: userInput }]);
    setIsLoading(true);
    
    try {
      const response = await fetch(`${apiUrl}/chat`, { 
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userInput }),
      });
      const result = await response.json();
      const botMessage = result.error ? { author: 'bot', content: result.error } : { author: 'bot', data: result };
      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      setMessages(prev => [...prev, { author: 'bot', content: "Sorry, I'm having trouble connecting." }]);
    } finally {
      setIsLoading(false);
      setUserInput('');
    }
  };
  
  const handleChatClick = (e) => {
      if (e.target.classList.contains('book-detail-link')) {
          e.preventDefault();
          const title = e.target.dataset.title;
          fetchBookDetails(title);
      }
  };

  return (
    <div className="chat-container">
      <div className="chat-header">
        <Link to="/" className="back-button">Return to Home Page!</Link>
      </div>

      <div className="chat-window" ref={chatWindowRef} onClick={handleChatClick}>
        {messages.map((msg, index) => (
          <div key={index} className={`message ${msg.author === 'user' ? 'user-message' : 'bot-message'}`}>
            <BotMessageContent message={msg} />
          </div>
        ))}
        {isLoading && <div className="message bot-message">Typing...</div>}
      </div>

      <form onSubmit={handleSendMessage} className="input-form">
        <input
          type="text"
          value={userInput}
          onChange={(e) => setUserInput(e.target.value)}
          disabled={isLoading}
          placeholder="Enter a book title or ask for details..."
          className="message-input"
        />
        <button type="submit" disabled={isLoading} className="send-button">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style={{ transform: 'rotate(0deg)' }}>
            <path d="M7 11L12 6L17 11M12 18V7" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        </button>
      </form>
    </div>
  );
};

export default Main;
