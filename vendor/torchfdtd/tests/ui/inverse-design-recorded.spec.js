import {test,expect} from '@playwright/test';
import fs from 'node:fs';

test('recorded boundary-history inverse design can plan, export and optimize',async({page})=>{
 test.setTimeout(120000);
 const errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.addInitScript(()=>{
  localStorage.setItem('torchfdtd.periodicDesign.v1',JSON.stringify({device:'cpu',iterations:1,steps:160,quadrature_counts:[4,4],initial_density:[[.2,.4],[.5,.3]]}));
  localStorage.removeItem('torchfdtd.periodicDesign.v1.job');
 });
 await page.goto('/');await page.getByRole('button',{name:'Inverse design',exact:true}).click();
 const panel=page.locator('.inverse-design-dialog');
 await panel.getByLabel('Execution mode',{exact:true}).selectOption('recorded');
 await expect(panel.locator('[data-recorded-scope]')).toContainText('Fields remain on the compute device');
 await expect(panel.getByLabel('Checkpoints',{exact:true})).toHaveCount(0);
 await panel.getByLabel('Transfer block (steps)',{exact:true}).fill('7');
 await panel.getByRole('button',{name:'Check memory',exact:true}).click();
 await expect(panel.locator('[data-plan-summary]')).toContainText('recorded');
 const download=page.waitForEvent('download');
 await panel.getByRole('button',{name:'Export Python',exact:true}).click();
 const source=fs.readFileSync(await (await download).path(),'utf8');
 expect(source).toContain("'execution': 'recorded'");
 expect(source).toContain("'recorded_trace_chunk_steps': 7");
 await panel.getByRole('button',{name:'Run inverse design',exact:true}).click();
 await expect(panel.locator('[data-design-status]')).toContainText('completed',{timeout:90000});
 const response=await page.request.get(await panel.locator('[data-design-download]').getAttribute('href'));
 const result=await response.json();
 expect(result.updates_completed).toBe(1);
 expect(result.history[0].gradient_l2).toBeGreaterThan(0);
 expect(result.history.every(row=>Number.isFinite(row.objective))).toBe(true);
 expect(result.config.execution).toBe('recorded');expect(result.plan.execution).toBe('recorded');
 expect(result.pending_density).toBeNull();
 await page.screenshot({path:'results/ui-inverse-design-recorded.png',fullPage:true});
 await panel.getByRole('button',{name:'Use evaluated result as new seed',exact:true}).click();
 await expect(panel.getByLabel('Execution mode',{exact:true})).toHaveValue('recorded');
 await expect(panel.getByLabel('Transfer block (steps)',{exact:true})).toHaveValue('7');
 expect(errors).toEqual([]);
});
