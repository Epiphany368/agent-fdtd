import {test,expect} from '@playwright/test';
import fs from 'node:fs';

test('edit six tensor coefficients, reject invalid material, save and run CPU scene',async({page})=>{
 const p=await (await page.request.get('/api/examples/3d')).json();
 p.region={...p.region,backend:'cpu',size:[1.2,1.2,1.2],mesh:.2,steps:20,snapshot_interval:5,
  boundaries:Object.fromEntries(['x','y','z'].flatMap(a=>['min','max'].map(s=>[a+'_'+s,{kind:'periodic'}])))};
 p.structures=[{id:'tensor',name:'Tensor inclusion',kind:'sphere',radius:.35,material:'SiN (constant n)'}];
 p.sources=[{id:'e',name:'Electric source',kind:'point',component:'Ez',center:[-.2,0,0],pulse:'continuous'}];
 p.monitors=[{id:'m',name:'Electric monitor',kind:'point',component:'Ez',center:[.2,0,0]}];
 await page.addInitScript(p=>{localStorage.setItem('torchfdtd.project.v1',JSON.stringify(p));localStorage.removeItem('torchfdtd.activeJob');},p);
 const errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto('/');await expect(page.locator('#tree')).toContainText('Tensor inclusion');
 await page.locator('[data-action="materials"]').click();
 const dialog=page.locator('.material-dialog');
 await dialog.getByLabel('Material list',{exact:true}).selectOption({label:'SiN (constant n)'});
 await dialog.getByLabel('Material model',{exact:true}).selectOption('tensor');
 for(const [k,v] of Object.entries({xx:2.2,yy:2.6,zz:3,xy:.12,xz:.08,yz:-.1})){
  await dialog.getByLabel('Tensor epsilon '+k,{exact:true}).fill(String(v));
  await dialog.getByLabel('Tensor epsilon '+k,{exact:true}).press('Tab');
 }
 await expect(dialog.locator('[data-preview]')).toBeDisabled();
 await expect(dialog.locator('.material-range')).toBeHidden();
 await expect(dialog.locator('.material-status')).toContainText('six Cartesian coefficients');
 const before=await page.evaluate(()=>localStorage.getItem('torchfdtd.project.v1'));
 await dialog.getByLabel('Tensor epsilon xx',{exact:true}).fill('-1');
 await dialog.getByLabel('Tensor epsilon xx',{exact:true}).press('Tab');
 await dialog.getByRole('button',{name:'Apply materials',exact:true}).click();
 await expect(dialog).toBeVisible();await expect(dialog.locator('.material-status')).toContainText('eigenvalues');
 expect(await page.evaluate(()=>localStorage.getItem('torchfdtd.project.v1'))).toBe(before);
 await dialog.getByLabel('Tensor epsilon xx',{exact:true}).fill('2.2');
 await dialog.getByLabel('Tensor epsilon xx',{exact:true}).press('Tab');
 await dialog.getByRole('button',{name:'Use supported tensor sampling',exact:true}).click();
 expect(await page.evaluate(()=>localStorage.getItem('torchfdtd.project.v1'))).toBe(before);
 await page.screenshot({path:'.local/tensor-ui/material-editor.png',fullPage:true});
 await dialog.getByRole('button',{name:'Apply materials',exact:true}).click();await expect(dialog).not.toBeVisible();
 const save=page.waitForEvent('download');await page.locator('[data-action="save"]').click();
 const saved=JSON.parse(fs.readFileSync(await (await save).path(),'utf8'));
 expect(saved.materials.find(m=>m.name==='SiN (constant n)').epsilon_tensor).toEqual([2.2,2.6,3,.12,.08,-.1]);
 expect(saved.region.material_sampling).toBe('yee');
 const submitted=page.waitForResponse(r=>r.url().endsWith('/api/jobs')&&r.request().method()==='POST');
 await page.locator('#run-button').click();const id=(await (await submitted).json()).id;
 await expect(page.locator('#mode-badge')).toHaveText('ANALYSIS',{timeout:30000});
 const job=await (await page.request.get('/api/jobs/'+id)).json();
 expect(job.status).toBe('completed');expect(job.summary.backend).toBe('cpu');
 expect(job.summary.engine.toLowerCase()).toContain('tensor');
 expect(job.monitors[0].signal.some(x=>Math.abs(x)>1e-12)).toBeTruthy();
 await page.screenshot({path:'.local/tensor-ui/completed.png',fullPage:true});
 expect(errors).toEqual([]);
});
