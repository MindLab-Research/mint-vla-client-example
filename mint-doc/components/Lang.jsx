'use client'

import { createContext, useContext, useState, useEffect } from 'react'

const LanguageContext = createContext({ lang: 'en', toggle: () => {} })

// 侧边栏翻译映射
const sidebarTranslations = {
  "Introduction": "简介",
  "Getting Started": "快速开始",
  "Using the API": "使用指南",
  "API Reference": "API 参考",
  "Installation": "安装",
  "Navigating These Docs": "文档导航",
  "Training and Sampling": "训练与采样",
  "Loss Functions": "损失函数",
  "Saving and Loading": "保存与加载",
  "Downloading Weights": "下载权重",
  "Publishing Weights": "发布权重",
  "Async and Futures": "异步操作",
  "Model Lineup": "支持的模型",
  "Parameters": "参数类型",
  "Exceptions": "异常"
}

// 翻译侧边栏
function translateSidebar(lang) {
  // 使用精确选择器：侧边栏中的 nextra-focus 链接
  const links = document.querySelectorAll('aside a.nextra-focus')
  links.forEach(link => {
    const text = link.textContent?.trim()
    if (!text) return

    // 避免重复翻译：使用 data-en 存储原始英文
    const enText = link.getAttribute('data-en') || text

    if (lang === 'zh' && sidebarTranslations[enText]) {
      if (!link.getAttribute('data-en')) {
        link.setAttribute('data-en', enText)
      }
      link.textContent = sidebarTranslations[enText]
    } else if (lang === 'en' && link.getAttribute('data-en')) {
      link.textContent = link.getAttribute('data-en')
    }
  })
}

// Check if text contains Chinese characters
function containsChinese(text) {
  return /[\u4e00-\u9fff]/.test(text)
}

// Filter TOC entries based on language
function filterTocEntries(lang) {
  const tocLinks = document.querySelectorAll('.nextra-toc a[href^="#"]')
  tocLinks.forEach(link => {
    const href = decodeURIComponent(link.getAttribute('href') || '')
    const isChinese = containsChinese(href)
    const li = link.closest('li')
    if (li) {
      // Show if language matches: Chinese href for zh, non-Chinese href for en
      const shouldShow = (lang === 'zh' && isChinese) || (lang === 'en' && !isChinese)
      li.style.display = shouldShow ? '' : 'none'
    }
  })
}

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

  // Filter TOC entries and translate sidebar when language changes
  useEffect(() => {
    if (mounted) {
      filterTocEntries(lang)
      translateSidebar(lang)
    }
  }, [lang, mounted])

  // Re-filter on route change (Nextra rebuilds TOC on navigation)
  useEffect(() => {
    if (!mounted) return
    const observer = new MutationObserver(() => {
      filterTocEntries(lang)
    })
    const toc = document.querySelector('.nextra-toc')
    if (toc) {
      observer.observe(toc, { childList: true, subtree: true })
    }
    return () => observer.disconnect()
  }, [lang, mounted])

  // Re-translate sidebar on route change with debounce
  useEffect(() => {
    if (!mounted) return

    let timeoutId = null
    const observer = new MutationObserver(() => {
      if (timeoutId) clearTimeout(timeoutId)
      timeoutId = setTimeout(() => translateSidebar(lang), 50)
    })

    // 延迟查找，确保 DOM 已渲染
    const timer = setTimeout(() => {
      const sidebar = document.querySelector('aside')
      if (sidebar) {
        observer.observe(sidebar, { childList: true, subtree: true })
      }
      translateSidebar(lang)
    }, 100)

    return () => {
      clearTimeout(timer)
      if (timeoutId) clearTimeout(timeoutId)
      observer.disconnect()
    }
  }, [lang, mounted])

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

export function NavTitle({ en, zh }) {
  const { lang } = useLanguage()
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
  }, [])

  if (!mounted) return en
  return lang === 'zh' ? zh : en
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
