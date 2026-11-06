import {test,expect} from '@playwright/test';

test('the feature checklist shows the combination registry with rule tooltips',async({page})=>{
 const errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto('/');await expect(page.locator('#tree')).toContainText('waveguide');
 await page.locator('[data-action="capabilities"]').click();
 const dialog=page.locator('.capability-dialog');
 await expect(dialog.locator('.capability-combinations summary')).toContainText('axis combinations run');
 const grid=dialog.locator('.combination-grid');
 await expect(grid.locator('.combination-cell.implemented').first()).toContainText('✓');
 await expect(grid.locator('.combination-cell.missing').first()).toContainText('✗');
 await dialog.getByLabel('Combination rows',{exact:true}).selectOption('material');
 await dialog.getByLabel('Combination columns',{exact:true}).selectOption('execution');
 await expect(grid.locator('.combination-head').first()).toContainText('Material / Execution');
 const rejected=grid.locator('.combination-cell.missing').first();
 expect(await rejected.getAttribute('title')).toContain('torchfdtd/');
 await page.screenshot({path:'results/ui-capability-combinations.png',fullPage:true});
 await dialog.getByRole('button',{name:'Close',exact:true}).click();
 expect(errors).toEqual([]);
});
