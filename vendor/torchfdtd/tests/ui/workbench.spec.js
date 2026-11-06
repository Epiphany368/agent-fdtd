import { test, expect } from '@playwright/test';

test('edit, drag, undo, save, run and inspect a real FDTD job',async({page})=>{
 const errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto('/');await expect(page.locator('#tree')).toContainText('waveguide');
 await page.locator('[data-select="waveguide"]').click();
 await page.getByLabel('y span',{exact:true}).fill('0.8');await page.getByLabel('y span',{exact:true}).press('Tab');
 await expect(page.getByLabel('y span',{exact:true})).toHaveValue('0.8');
 await page.locator('[data-action="duplicate"]').click();await expect(page.locator('#tree')).toContainText('waveguide_copy');
 await page.locator('[data-action="delete"]').click();await expect(page.locator('#tree')).not.toContainText('waveguide_copy');
 await page.locator('[data-action="undo"]').last().click();await expect(page.locator('#tree')).toContainText('waveguide_copy');
 await page.locator('#tree .tree-row').filter({hasText:'waveguide_copy'}).click();await page.locator('[data-action="delete"]').click();
 await page.locator('[data-add="circle"]').click();
 await page.getByLabel('x',{exact:true}).fill('0.5');await page.getByLabel('x',{exact:true}).press('Tab');
 const canvas=page.locator('.xy canvas'),rect=await canvas.boundingBox();
 await page.mouse.move(rect.x+rect.width/2+rect.width*.045,rect.y+rect.height/2);await page.mouse.down();await page.mouse.move(rect.x+rect.width/2+rect.width*.1,rect.y+rect.height/2-25,{steps:8});await page.mouse.up();
 await expect(page.getByLabel('y',{exact:true})).not.toHaveValue('0');
 await page.locator('[data-action="save"]').click();
 await page.locator('[data-action="python"]').click();await expect(page.locator('#python-editor')).toContainText('');await expect(page.locator('#python-editor')).toHaveValue(/Simulation\(project\).run/);
 await page.locator('[data-select="fdtd"]').click();await page.getByLabel('resource',{exact:true}).selectOption(process.env.TORCHFDTD_TEST_CUDA?'cuda':'cpu');await page.getByLabel('time steps',{exact:true}).fill('300');await page.getByLabel('time steps',{exact:true}).press('Tab');
 await page.screenshot({path:'results/ui-layout.png',fullPage:true});
 await page.locator('#run-button').click();await expect(page.locator('#mode-badge')).toHaveText('ANALYSIS',{timeout:45000});
 await expect(page.locator('#results-tree')).toContainText('field snapshots');await expect(page.locator('#run-button')).toBeDisabled();
 if(process.env.TORCHFDTD_TEST_CUDA){await expect(page.locator('#connection')).toContainText('5880');await expect(page.locator('#results-tree')).toContainText('CUDA graph');}
 await expect(page.locator('[data-add="rectangle"]')).toBeDisabled();
 await page.locator('[data-plot="spectrum"]').click();await page.screenshot({path:'results/ui-results.png',fullPage:true});
 await page.locator('#layout-button').click();await expect(page.locator('#run-button')).toBeEnabled();
 expect(errors).toEqual([]);
});
