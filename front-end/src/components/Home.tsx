import RegisterForm from './RegisterForm';

const Home = () => {
  return (
    <div className="home-container">
      <h1>Welcome to Our Web App</h1>
      <p>Register or login to continue.</p>
      <RegisterForm />
      <style jsx>{`
        .home-container {
          text-align: center;
          padding: 20px;
        }
        h1 {
          color: #333;
        }
        p {
          color: #666;
        }
      `}</style>
    </div>
  );
};

export default Home;
