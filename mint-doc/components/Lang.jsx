'use client'

import { createContext, useContext, useState, useEffect } from 'react'

const LanguageContext = createContext({ lang: 'en', toggle: () => {} })

export function LanguageProvider({ children }) {
  const [lang, setLang] = useState('en')
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
    const saved = localStorage.getItem('mint-doc-lang')
    if (saved === 'en' || saved === 'zh') {
      setLang(saved)
    }
  }, [])

  const toggle = () => {
    const next = lang === 'en' ? 'zh' : 'en'
    setLang(next)
    localStorage.setItem('mint-doc-lang', next)
  }

  // Prevent hydration mismatch by rendering default during SSR
  if (!mounted) {
    return (
      <LanguageContext.Provider value={{ lang: 'en', toggle }}>
        {children}
      </LanguageContext.Provider>
    )
  }

  return (
    <LanguageContext.Provider value={{ lang, toggle }}>
      {children}
    </LanguageContext.Provider>
  )
}

export function useLanguage() {
  return useContext(LanguageContext)
}

export function En({ children }) {
  const { lang } = useLanguage()
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
  }, [])

  // During SSR and initial mount, show English (default)
  if (!mounted) return <>{children}</>

  return lang === 'en' ? <>{children}</> : null
}

export function Zh({ children }) {
  const { lang } = useLanguage()
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
  }, [])

  // During SSR and initial mount, hide Chinese
  if (!mounted) return null

  return lang === 'zh' ? <>{children}</> : null
}

export function LanguageSwitcher() {
  const { lang, toggle } = useLanguage()
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
  }, [])

  if (!mounted) {
    return (
      <button
        style={{
          background: 'none',
          border: '1px solid currentColor',
          borderRadius: '4px',
          padding: '4px 8px',
          cursor: 'pointer',
          fontSize: '14px',
          color: 'inherit',
          opacity: 0.8
        }}
      >
        中文
      </button>
    )
  }

  return (
    <button
      onClick={toggle}
      style={{
        background: 'none',
        border: '1px solid currentColor',
        borderRadius: '4px',
        padding: '4px 8px',
        cursor: 'pointer',
        fontSize: '14px',
        color: 'inherit',
        opacity: 0.8
      }}
      title={lang === 'en' ? 'Switch to Chinese' : '切换到英文'}
    >
      {lang === 'en' ? '中文' : 'English'}
    </button>
  )
}
