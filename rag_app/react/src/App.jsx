import { useState } from 'react'
import Login from './components/Login'
import Chat from './components/Chat'

function App() {
  const [loggedInUser, setLoggedInUser] = useState(null)

  if (!loggedInUser) {
    return <Login onLogin={setLoggedInUser} />
  }

  return <Chat user={loggedInUser} onLogout={() => setLoggedInUser(null)} />
}

export default App
