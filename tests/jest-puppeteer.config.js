module.exports = {
    launch: {
      headless: 'new',
    //   headless: true,
      defaultViewport: {
        width: 1600,
        height: 900
      },
      // defaultViewport: null, // Disable the default viewport
      // args: ['--start-maximized'],
      timeout: 120000
    },
    exitOnPageError: false,
  }