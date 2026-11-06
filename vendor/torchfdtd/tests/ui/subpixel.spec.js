import {test,expect} from '@playwright/test';

test('choose experimental dielectric interfaces, retain Python settings and run',async({page})=>{
 const errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto('/');await expect(page.locator('#tree')).toContainText('waveguide');
 await page.locator('[data-select="fdtd"]').click();
 await page.getByLabel('interface method',{exact:true}).selectOption('subpixel');
 await expect(page.getByLabel('interface sampling',{exact:true})).toHaveValue('yee');
 await expect(page.locator('#properties')).toContainText('Lossless dielectrics');
 const order=page.getByLabel('face quadrature order',{exact:true});await order.fill('16');await order.press('Tab');
 await page.getByLabel('time steps',{exact:true}).fill('100');await page.getByLabel('time steps',{exact:true}).press('Tab');
 await page.getByLabel('resource',{exact:true}).selectOption(process.env.TORCHFDTD_TEST_CUDA?'cuda':'cpu');
 await page.locator('[data-action="python"]').click();
 await expect(page.locator('#python-editor')).toHaveValue(/interface_method.*subpixel/s);
 const saved=JSON.parse(await page.evaluate(()=>localStorage.getItem('torchfdtd.project.v1')));
 expect(saved.region.interface_method).toBe('subpixel');expect(saved.region.subpixel_quadrature).toBe(16);
 await page.locator('#run-button').click();await expect(page.locator('#mode-badge')).toHaveText('ANALYSIS',{timeout:90000});
 await page.locator('#layout-button').click();
 await page.getByLabel('interface method',{exact:true}).selectOption('staircase');
 await expect(page.getByLabel('face quadrature order',{exact:true})).toHaveCount(0);
 expect(errors).toEqual([]);
});
