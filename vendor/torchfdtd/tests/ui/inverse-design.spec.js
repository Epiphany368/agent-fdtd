import {test,expect} from '@playwright/test';
import fs from 'node:fs';

test('periodic density editor, memory plan, real adjoint update and Python export',async({page})=>{
 test.setTimeout(120000);
 const errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.addInitScript(()=>{
  localStorage.setItem('torchfdtd.periodicDesign.v1',JSON.stringify({device:'cpu',iterations:1,steps:80,quadrature_counts:[4,4],initial_density:[[.2,.4],[.6,.3]]}));
  localStorage.removeItem('torchfdtd.periodicDesign.v1.job');
 });
 await page.goto('/');await expect(page.locator('#tree')).toContainText('FDTD');
 await page.getByRole('button',{name:'Inverse design',exact:true}).click();
 const panel=page.locator('.inverse-design-dialog');
 await expect(panel).toBeVisible();
 await panel.getByRole('button',{name:'Check memory',exact:true}).click();
 await expect(panel.locator('[data-plan-summary]')).toContainText('resident');
 await panel.getByLabel('Density paint value',{exact:true}).fill('0.7');
 await panel.getByLabel('Density editor',{exact:true}).click({position:{x:40,y:40}});
 const savedDownload=page.waitForEvent('download');
 await panel.getByRole('button',{name:'Save setup JSON',exact:true}).click();
 const saved=JSON.parse(fs.readFileSync(await (await savedDownload).path(),'utf8'));
 expect(saved.initial_density[0][1]).toBe(.7);
 const pythonDownload=page.waitForEvent('download');
 await panel.getByRole('button',{name:'Export Python',exact:true}).click();
 const python=fs.readFileSync(await (await pythonDownload).path(),'utf8');
 expect(python).toContain('run_periodic_design');expect(python).toContain("'initial_density': [[0.2, 0.7], [0.6, 0.3]]");
 await panel.getByRole('button',{name:'Run inverse design',exact:true}).click();
 await expect(panel.locator('[data-design-status]')).toContainText('completed',{timeout:90000});
 await expect(panel.locator('[data-design-history] tr')).toHaveCount(2);
 const response=await page.request.get(await panel.locator('[data-design-download]').getAttribute('href'));
 const result=await response.json();expect(result.updates_completed).toBe(1);expect(result.history[0].gradient_l2).toBeGreaterThan(0);
 expect(result.last_evaluated.update).toBe(1);expect(result.pending_density).toBeNull();
 expect(result.last_evaluated.density.flat().every(v=>Number.isFinite(v)&&v>=0&&v<=1)).toBe(true);
 await page.screenshot({path:'results/ui-inverse-design.png',fullPage:true});
 await panel.getByRole('button',{name:'Use evaluated result as new seed',exact:true}).click();
 await expect(panel.locator('[data-density-caption]')).toContainText('Initial density');
 expect(errors).toEqual([]);
});
