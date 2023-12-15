module.exports = {
    launch: {
      headless: 'new',
    //   headless: true,
      // defaultViewport: {
      //   width: 1300,
      //   height: 1024
      // },
      defaultViewport: null, // Disable the default viewport
      args: ['--start-maximized'],
      timeout: 120000
    },
    exitOnPageError: false,
  }