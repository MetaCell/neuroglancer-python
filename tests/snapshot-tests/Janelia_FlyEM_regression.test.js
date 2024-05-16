//IMPORTS:
import "expect-puppeteer";
const { toMatchImageSnapshot } = require('jest-image-snapshot');
expect.extend({ toMatchImageSnapshot });
const path = require("path");
var scriptName = path.basename(__filename, ".js");
import * as selectors from "./selectors.js";
import { THRESHOLD, URL_START, TIMEOUT } from "./constants.js";

//PAGE INFO:
const baseURL = process.env.url || `${URL_START}#!%7B%22dimensions%22:%7B%22x%22:%5B8e-9%2C%22m%22%5D%2C%22y%22:%5B8e-9%2C%22m%22%5D%2C%22z%22:%5B8e-9%2C%22m%22%5D%7D%2C%22position%22:%5B3223.5%2C3321.5%2C4045.5%5D%2C%22crossSectionScale%22:1%2C%22projectionScale%22:8192%2C%22layers%22:%5B%7B%22type%22:%22image%22%2C%22source%22:%22precomputed://gs://neuroglancer-public-data/flyem_fib-25/image%22%2C%22tab%22:%22source%22%2C%22shaderControls%22:%7B%22colormap%22:%7B%22controlPoints%22:%5B%7B%22position%22:204%2C%22color%22:%7B%220%22:0%2C%221%22:0%2C%222%22:0%2C%223%22:0%7D%7D%2C%7B%22position%22:358%2C%22color%22:%7B%220%22:255%2C%221%22:255%2C%222%22:255%2C%223%22:255%7D%7D%5D%7D%7D%2C%22name%22:%22image%22%7D%5D%2C%22selectedLayer%22:%7B%22visible%22:true%2C%22layer%22:%22image%22%7D%2C%22layout%22:%224panel%22%7D`;


//SNAPSHOT:
const SNAPSHOT_OPTIONS = {
  customSnapshotsDir: `./snapshot-tests/snapshots/${scriptName}`,
  comparisonMethod: "ssim",
  failureThresholdType: "percent",
  failureThreshold: THRESHOLD,
};


//TESTS:

jest.setTimeout(TIMEOUT);


describe("Test Suite for Janelia FlyEM Dataset", () => {
  beforeAll(async () => {
    
    await page.goto(baseURL);
    await page.waitForTimeout(3000);
    await page.waitForSelector(selectors.SIDE_PANEL);

  });

  
  describe("2D Canvas", () => {

    it("should navigate to rendering tab", async () => {
      console.log('Navigating to Rendering tab ...')
      await page.waitForSelector(selectors.SIDE_PANEL_TAB)
      await page.evaluate(() => {
        [...document.querySelectorAll('.neuroglancer-tab-label')].find(element => element.innerText === 'Rendering').click();
      });
      await page.waitForTimeout(1000);
      await page.waitForSelector(selectors.RENDERING_TAB_CONTROLS)
      const rendering_options = await page.$$(".neuroglancer-layer-control-container.neuroglancer-layer-options-control-container");
      expect(rendering_options.length).toBe(5);
      console.log('Tab reached')
    })

    it("should wait for resolution slices to be displayed", async () => {
      console.log('Waiting for resolution slices to load ...')
      try {
        const isValueComplete = async () => {
          const { valueBeforeSlash, valueAfterSlash } = await page.$eval('div[title="Number of chunks rendered"]', (element) => {
            const textContent = element.textContent.trim();
            const [valueBeforeSlash, valueAfterSlash] = textContent.split('/').map(part => part.trim());
            
            return { valueBeforeSlash, valueAfterSlash };
          });
          return valueBeforeSlash >= (1 / 2) * valueAfterSlash;
        };

        const maxRetries = 20;
        let retries = 0;
    
        while (retries < maxRetries) {
          if (await isValueComplete()) {
            console.log('Value is reached. Continuing with the next steps.');
            break;
          }
          await page.waitForTimeout(3000); 
          retries++;
        }
    
        if (retries === maxRetries) {
          throw new Error('Timeout: Value did not become the expected within the specified time.');
        }
      } catch (error) {
        console.error('An error occurred:', error.message);
        throw error;
      }

    })


    it("should take screenshot of main canvas", async () => {
      const canvas = await page.waitForSelector(selectors.IMAGE_CANVAS, {hidden:false});
      await page.waitForTimeout(1000 * 6);
      const groups_image = await page.screenshot();
      // const groups_image = await canvas.screenshot();
      await console.log("... taking canvas snapshot ...");
      expect(groups_image).toMatchImageSnapshot({
        ...SNAPSHOT_OPTIONS,
        customSnapshotIdentifier: 'Main_canvas',
      });
      await page.waitForTimeout(1000 * 3);
    });
  })

  describe("Canvas with Volume Rendering", () => {

    it("should enable volume rendering", async () => {
      console.log('Enabling Volume Rendering ...')
      await page.waitForSelector(selectors.RENDERING_TAB_DROPDOWNS)
      const dropdown_buttons = await page.$$('select.neuroglancer-layer-control-control')
        await dropdown_buttons[1].click()
      await page.waitForSelector(selectors.OFF_VALUE)
      await page.waitForSelector(selectors.ON_VALUE)
      await page.waitForSelector(selectors.MAX_VALUE)
      await dropdown_buttons[1].select('on');
      await page.waitForTimeout(2000);
      await page.waitForSelector(selectors.RESOLUTION_SLICES)
      const rendering_options_afterVolume = await page.$$(".neuroglancer-layer-control-container.neuroglancer-layer-options-control-container");
      expect(rendering_options_afterVolume.length).toBe(6);
      console.log('Volume Rendering enabled')
    });

    it("should wait for 3D resolution samples to be displayed", async () => {
      console.log('Waiting for 3D resolution samples to load ...')
      await page.waitForSelector(selectors.RESOLUTION_SAMPLES)
      try {
        const isValueComplete = async () => {
          const { valueBeforeSlash, valueAfterSlash } = await page.$eval('.neuroglancer-tab-content.neuroglancer-image-dropdown > div > .neuroglancer-layer-control-container.neuroglancer-layer-options-control-container > .neuroglancer-render-scale-widget.neuroglancer-layer-control-control > .neuroglancer-render-scale-widget-legend > div[title="Number of chunks rendered"]', (element) => {
            const textContent = element.textContent.trim();
            const [valueBeforeSlash, valueAfterSlash] = textContent.split('/').map(part => part.trim());
            
            return { valueBeforeSlash, valueAfterSlash };
          });
          
          return valueBeforeSlash >= (1 / 4) * valueAfterSlash;
        };

        const maxRetries = 20;
        let retries = 0;
    
        while (retries < maxRetries) {
          if (await isValueComplete()) {
            console.log('Value is reached. Continuing with the next steps.');
            break;
          }
          await page.waitForTimeout(3000); 
          retries++;
        }
    
        if (retries === maxRetries) {
          throw new Error('Timeout: Value did not become the expected within the specified time.');
        }
      } catch (error) {
        console.error('An error occurred:', error.message);
        throw error;
      }
    })

    
    it("should take screenshot of main canvas with 3D", async () => {
      const canvas = await page.waitForSelector(selectors.IMAGE_CANVAS, {hidden:false});
      await page.waitForTimeout(1000 * 6);
      const groups_image = await page.screenshot();
      // const groups_image = await canvas.screenshot();
      await console.log("... taking canvas snapshot ...");
      expect(groups_image).toMatchImageSnapshot({
        ...SNAPSHOT_OPTIONS,
        customSnapshotIdentifier: 'Main_canvas_w_3D',
      });
      await page.waitForTimeout(1000 * 3);
    });

  })

  
  describe("Canvas with Max Volume Rendering", () => {

    it("should enable max volume rendering", async () => {
      console.log('Enabling Max Volume Rendering ...')
      await page.waitForSelector(selectors.RENDERING_TAB_DROPDOWNS)
      const dropdown_buttons = await page.$$('select.neuroglancer-layer-control-control')
        await dropdown_buttons[1].click()
      await page.waitForSelector(selectors.OFF_VALUE)
      await page.waitForSelector(selectors.ON_VALUE)
      await page.waitForSelector(selectors.MAX_VALUE)
      await dropdown_buttons[1].select('max');
      await page.waitForTimeout(2000);
      await page.waitForSelector(selectors.RESOLUTION_SLICES)
      
      console.log('Max Volume Rendering enabled')
    })

    it("should take screenshot of main canvas with Max 3D Rendering", async () => {
      const canvas = await page.waitForSelector(selectors.IMAGE_CANVAS, {hidden:false});
      await page.waitForSelector('button[title="Switch to 3d layout."]', {hidden:false});
      await page.click('button[title="Switch to 3d layout."]');
      await page.waitForSelector('button[title="Switch to 4panel layout."]', {hidden:false});

      // await page.waitForTimeout(1000 * 6);
      const groups_image = await page.screenshot();
      // const groups_image = await canvas.screenshot();
      await console.log("... taking canvas snapshot ...");
      expect(groups_image).toMatchImageSnapshot({
        ...SNAPSHOT_OPTIONS,
        customSnapshotIdentifier: 'Max_3D_Rendering',
      });
      await page.waitForTimeout(1000 * 3);
      await page.waitForSelector('button[title="Switch to 4panel layout."]', {hidden:false});
      await page.click('button[title="Switch to 4panel layout."]');
      await page.waitForSelector('button[title="Switch to 3d layout."]', {hidden:false});

    });

  });

  describe("Canvas with Min Volume Rendering", () => {

    it("should enable min volume rendering", async () => {
      console.log('Enabling Min Volume Rendering ...')
      await page.waitForSelector(selectors.RENDERING_TAB_DROPDOWNS)
      const dropdown_buttons = await page.$$('select.neuroglancer-layer-control-control')
        await dropdown_buttons[1].click()
      await page.waitForSelector(selectors.OFF_VALUE)
      await page.waitForSelector(selectors.ON_VALUE)
      await page.waitForSelector(selectors.MAX_VALUE)
      await page.waitForSelector(selectors.MIN_VALUE)
      await dropdown_buttons[1].select('min');
      await page.waitForTimeout(2000);
      await page.waitForSelector(selectors.RESOLUTION_SLICES)
      
      console.log('Min Volume Rendering enabled')
    })

    it("should take screenshot of main canvas with Min 3D Rendering", async () => {
      const canvas = await page.waitForSelector(selectors.IMAGE_CANVAS, {hidden:false});
      await page.waitForSelector('button[title="Switch to 3d layout."]', {hidden:false});
      await page.click('button[title="Switch to 3d layout."]');
      await page.waitForSelector('button[title="Switch to 4panel layout."]', {hidden:false});

      // await page.waitForTimeout(1000 * 6);
      const groups_image = await page.screenshot();
      // const groups_image = await canvas.screenshot();
      await console.log("... taking canvas snapshot ...");
      expect(groups_image).toMatchImageSnapshot({
        ...SNAPSHOT_OPTIONS,
        customSnapshotIdentifier: 'Min_3D_Rendering',
      });
      await page.waitForTimeout(1000 * 3);
      await page.waitForSelector('button[title="Switch to 4panel layout."]', {hidden:false});
      await page.click('button[title="Switch to 4panel layout."]');
      await page.waitForSelector('button[title="Switch to 3d layout."]', {hidden:false});

    });
  })

  describe.skip("Canvas with colored 2D + 3D", () => {

    it("should change the color map of the 3D rendering", async () => {
      console.log('Changing color map ...')
      await page.waitForSelector(selectors.COLORMAP)
      await page.waitForSelector(selectors.COLORMAP_COLOR_WIDGET)
      await page.$eval('#neuroglancer-tf-color-widget', (colorWidget) => {
        colorWidget.value = '#00ff00'; 
        const event = new Event('change', { bubbles: true });
        colorWidget.dispatchEvent(event);
      });
      await page.waitForTimeout(1000 * 3);

      await page.click(selectors.COLORMAP_KNEES)
      await page.waitForTimeout(1000 * 5);
      await page.waitForSelector(selectors.RENDERING_TAB_DROPDOWNS)
      const dropdown_buttons = await page.$$('select.neuroglancer-layer-control-control')
        await dropdown_buttons[1].click()
      await page.waitForSelector(selectors.OFF_VALUE)
      await page.waitForSelector(selectors.ON_VALUE)
      await page.waitForSelector(selectors.MAX_VALUE)
      await dropdown_buttons[1].select('on');

      console.log('Color map changed')
    })

    it("should take screenshot of main colored canvas with 3D", async () => {
      const canvas = await page.waitForSelector(selectors.IMAGE_CANVAS, {hidden:false});
      await page.waitForTimeout(1000 * 6);
      const groups_image = await page.screenshot();
      // const groups_image = await canvas.screenshot();
      await console.log("... taking canvas snapshot ...");
      expect(groups_image).toMatchImageSnapshot({
        ...SNAPSHOT_OPTIONS,
        customSnapshotIdentifier: 'colored_canvas',
      });
      await page.waitForTimeout(1000 * 3);
    });

  })

});
