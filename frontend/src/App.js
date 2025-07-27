import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Main from './components/Main';
import Intro from './components/Intro'; 

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Intro />} />
        <Route path="/chatbot" element={<Main />} />
      </Routes>
    </Router>
  );
}

export default App;
