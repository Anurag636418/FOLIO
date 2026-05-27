import { useState, useRef, useEffect } from 'react'
import { Plus, Send, FileText, BookOpen } from 'lucide-react'
import { uploadFile, chatStream } from './api'
import ReactMarkdown from 'react-markdown'
import './index.css'

interface Message {
  role: 'user' | 'assistant'
  content: string
  sources?: any[]
}

function App() {
  const [page, setPage] = useState<'landing' | 'app'>('landing')
  const [docs, setDocs] = useState<string[]>([])
  const [activeDoc, setActiveDoc] = useState<string | null>(null)
  const [sessionTokens, setSessionTokens] = useState<Record<string, string>>({})
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [isUploading, setIsUploading] = useState(false)
  const [isStreaming, setIsStreaming] = useState(false)
  
  const fileInputRef = useRef<HTMLInputElement>(null)
  const chatEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    setIsUploading(true)
    try {
      const res = await uploadFile(file)
      if (!docs.includes(res.filename)) setDocs([...docs, res.filename])
      setActiveDoc(res.filename)
      setSessionTokens(prev => ({ ...prev, [res.filename]: res.session_token }))
      setMessages([])
    } catch (err: any) {
      alert("Error: " + err.message)
    } finally {
      setIsUploading(false)
      if (fileInputRef.current) fileInputRef.current.value = ''
    }
  }

  const handleSend = async () => {
    if (!input.trim() || !activeDoc || isStreaming) return
    const query = input
    setInput('')
    
    setMessages(prev => [...prev, { role: 'user', content: query }])
    setIsStreaming(true)
    
    try {
      let fullResponse = ''
      setMessages(prev => [...prev, { role: 'assistant', content: '' }])
      
      const generator = chatStream(query, activeDoc, sessionTokens[activeDoc] ?? "", messages)
      for await (const chunk of generator) {
        if (chunk.error) {
          fullResponse = chunk.error
        } else if (chunk.done) {
          setMessages(prev => {
            const arr = [...prev]
            arr[arr.length - 1].sources = chunk.sources
            return arr
          })
          break
        } else if (chunk.text) {
          fullResponse += chunk.text
        }
        
        setMessages(prev => {
          const arr = [...prev]
          arr[arr.length - 1].content = fullResponse
          return arr
        })
      }
    } catch (err) {
      console.error(err)
    } finally {
      setIsStreaming(false)
    }
  }

  if (page === 'landing') {
    return (
      <div className="landing-page">
        <header className="landing-header">
          <div className="landing-logo">
            <BookOpen strokeWidth={2.5} size={24} color="#000" /> Folio
          </div>
        </header>

        <main className="landing-main">
          <h1 className="landing-title">
            Talk to your <span className="gradient-text">Documents</span>
          </h1>
          <p className="landing-subtitle">
            Upload your PDFs, Word files, and text documents. Instantly extract insights and ask questions, completely grounded in your own data.
          </p>
          <button className="start-btn" onClick={() => setPage('app')}>
            Open Folio
          </button>
        </main>
      </div>
    )
  }

  return (
    <div className="app-container">
      {/* SOURCE BOARD */}
      <div className="source-board">
        <div className="board-header">
          <BookOpen size={24} color="var(--primary-accent)" strokeWidth={2.5} />
          <span style={{ fontSize: '20px', letterSpacing: '-0.02em', color: '#1f1f1f' }}>Folio</span>
        </div>
        
        <div className="board-content">
          <input type="file" ref={fileInputRef} onChange={handleUpload} style={{ display: 'none' }} accept=".pdf,.docx,.txt" />
          <button className="upload-card" onClick={() => fileInputRef.current?.click()} disabled={isUploading}>
            <Plus size={20} />
            {isUploading ? 'Uploading...' : 'Add source'}
          </button>

          {docs.map(doc => (
            <div key={doc} className={`source-card ${doc === activeDoc ? 'active' : ''}`} onClick={() => { setActiveDoc(doc); setMessages([]) }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <FileText size={16} color={doc === activeDoc ? 'var(--primary-accent)' : '#444746'} />
                <span style={{ fontWeight: 600, fontSize: '14px', color: doc === activeDoc ? 'var(--primary-accent)' : '#1f1f1f', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                  {doc}
                </span>
              </div>
              <span style={{ fontSize: '12px', color: doc === activeDoc ? 'var(--primary-accent)' : '#747775' }}>Document</span>
            </div>
          ))}
        </div>
      </div>

      {/* CHAT PANEL */}
      <div className="chat-panel">
        <div className="chat-scroll">
          <div className="chat-inner">
            {!activeDoc ? (
              <div className="empty-chat">
                <h2>Ready to Research</h2>
                <p>Select or upload a document to begin generating insights.</p>
              </div>
            ) : messages.length === 0 ? (
              <div className="empty-chat">
                <h2>{activeDoc}</h2>
                <p>Notebook guide initialized. Ask anything about this document.</p>
              </div>
            ) : (
              messages.map((m, i) => (
                <div key={i} className={`message ${m.role}`}>
                  <div className="message-avatar">
                    {m.role === 'user' ? 'U' : 'F'}
                  </div>
                  <div className="message-content">
                    <ReactMarkdown>{m.content}</ReactMarkdown>
                    {m.sources && m.sources.length > 0 && (
                      <div style={{ marginTop: '8px', display: 'flex', flexWrap: 'wrap' }}>
                        {m.sources.map((s, si) => (
                          <span key={si} className="citation-pill">
                            <FileText size={12} /> Pg {s.page}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              ))
            )}
            <div ref={chatEndRef} />
          </div>
        </div>

        {activeDoc && (
          <div className="input-zone">
            <div className="input-zone-inner">
              <div className="input-wrapper">
                <textarea 
                  className="chat-input" 
                  placeholder="Ask your folio..." 
                  value={input}
                  onChange={e => setInput(e.target.value)}
                  onKeyDown={e => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault()
                      handleSend()
                    }
                  }}
                  rows={1}
                />
                <button className="send-btn" onClick={handleSend} disabled={!input.trim() || isStreaming}>
                  <Send size={18} />
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default App
