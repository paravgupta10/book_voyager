import React from 'react';
import { Link } from 'react-router-dom';
import './Intro.css'; 

const Intro = () => {
  return (
    <div className="intro-container">
      <h1 className="intro-heading">Book Voyager.AI</h1>
      <p className="intro-description">
        Discover your next great read through the power of AI!<br /> Our chatbot uses semantic search to find books based on the themes and styles of titles you already love.
      </p>
      <p className="intro-description">
        Simply enter a book title to get started, and click on any recommendation to get a short summary.
      </p>
      <div className="intro-button-container">
        <Link to="/chatbot" className="intro-button">
          Take me to the Chatbot!
        </Link>
      </div>
    </div>
  );
};

export default Intro;
