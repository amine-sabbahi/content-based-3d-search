// app/page.tsx
"use client";
import { useRouter } from 'next/navigation';

const MainPage = () => {
  const router = useRouter();
  
  return (
    <div className="main-container">
      <h1>Welcome to Our Web App</h1>
      <p>Please Register or Login to Continue</p>
      <div className="button-group">
        <button onClick={() => router.push('/register')}>Register</button>
        <button onClick={() => router.push('/login')}>Login</button>
      </div>
      <style jsx>{`
        .main-container {
          text-align: center;
          padding: 50px;
        }
        h1 {
          color: #333;
        }
        p {
          color: #666;
          margin-bottom: 20px;
        }
        .button-group {
          display: flex;
          justify-content: center;
          gap: 20px;
        }
        button {
          padding: 10px 20px;
          font-size: 16px;
          color: white;
          background-color: #0070f3;
          border: none;
          border-radius: 5px;
          cursor: pointer;
        }
        button:hover {
          background-color: #005bb5;
        }
      `}</style>
    </div>
  );
};

export default MainPage;