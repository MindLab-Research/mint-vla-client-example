export default {
  logo: <span>Mind Lab Toolkit (MinT)</span>,
  project: {
    link: 'https://github.com/MindLab-Research/mindlab-toolkit'
  },
  editLink: {
    component: null
  },
  feedback: {
    content: null
  },
  footer: {
    text: 'Mind Lab Toolkit (MinT) Documentation'
  },
  head: (
    <>
      <meta name="viewport" content="width=device-width, initial-scale=1.0" />
      <meta property="og:title" content="Mind Lab Toolkit (MinT)" />
      <meta property="og:description" content="Mind Lab Toolkit (MinT) - Training API for LLMs" />
    </>
  ),
  primaryHue: 210,
  primarySaturation: 100,
  useNextSeoProps() {
    return {
      titleTemplate: '%s – MinT'
    }
  }
}
