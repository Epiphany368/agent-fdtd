import {test,expect} from '@playwright/test';
import {execFileSync} from 'node:child_process';
import fs from 'node:fs';

test('edit FSP axis mesh, export, reimport and execute the edited GPU grid',async({page})=>{
 const errors=[];page.on('pageerror',e=>errors.push(e.message));
 const python=process.env.TORCHFDTD_TEST_PYTHON||(process.platform==='win32'?'.venv/Scripts/python.exe':'.venv/bin/python');
 const raw=Buffer.from(execFileSync(python,['-c',"import sys,base64;sys.path.insert(0,'tests');from test_fsp_settings_write import settings_fixture;print(base64.b64encode(settings_fixture()).decode())"],{encoding:'utf8'}).trim(),'base64');
 await page.goto('/');await expect(page.locator('#tree')).toContainText('waveguide');
 await page.locator('[data-action="fsp-native"]').click();
 await page.locator('#fsp-native-input').setInputFiles({name:'authored-mesh.fsp',mimeType:'application/octet-stream',buffer:raw});
 await expect(page.locator('#fsp-native-status')).toContainText('Ready to open',{timeout:20000});
 await page.locator('[data-native="load"]').click();
 await expect(page.locator('#fsp-native-dialog')).not.toBeVisible();
 await expect(page.getByLabel('time steps',{exact:true})).toHaveValue('20');
 await page.getByLabel('Independent axis spacing',{exact:true}).check();
 for(const [axis,value] of [['dx','0.12'],['dy','0.16'],['dz','0.2']]){
  await page.getByLabel(axis,{exact:true}).fill(value);await page.getByLabel(axis,{exact:true}).press('Tab');
 }
 await expect(page.getByLabel('interface sampling',{exact:true})).toHaveValue('yee');
 await page.getByLabel('time steps',{exact:true}).fill('80');await page.getByLabel('time steps',{exact:true}).press('Tab');
 await expect(page.locator('#mesh-summary')).toContainText('40 × 30 × 24');
 await page.locator('[data-action="fsp-native"]').click();await page.locator('[data-native="export"]').click();
 await expect(page.locator('#fsp-native-status')).toContainText('Scene export verified',{timeout:20000});
 await expect(page.locator('[data-native-mesh-export]')).toContainText('48 × 48 × 48 → 40 × 30 × 24');
 const download=page.waitForEvent('download');await page.locator('[data-native="edited"]').click();
 const bytes=fs.readFileSync(await (await download).path());expect(bytes.equals(raw)).toBe(false);
 await page.screenshot({path:'results/ui-fsp-mesh-export.png',fullPage:true});
 await page.locator('#fsp-native-input').setInputFiles({name:'edited-mesh.fsp',mimeType:'application/octet-stream',buffer:bytes});
 await expect(page.locator('#fsp-native-status')).toContainText('Ready to open',{timeout:20000});
 await page.locator('[data-native="load"]').click();await expect(page.locator('#fsp-native-dialog')).not.toBeVisible();
 await expect(page.locator('#mesh-summary')).toContainText('40 × 30 × 24');
 const p=await page.evaluate(()=>JSON.parse(localStorage.getItem('torchfdtd.project.v1')));
 expect(p.region.mesh_coordinates.map(a=>a.length)).toEqual([41,31,25]);
 expect(p.region.material_sampling).toBe('yee');expect(p.region.steps).toBe(80);
 const exported=await page.request.post('/api/python',{data:p});expect(await exported.text()).toContain('mesh_coordinates');
 await page.getByLabel('resource',{exact:true}).selectOption(process.env.TORCHFDTD_TEST_CUDA?'cuda':'cpu');
 const submitted=page.waitForResponse(r=>r.url().endsWith('/api/jobs')&&r.request().method()==='POST');
 await page.locator('#run-button').click();const job=await (await submitted).json();
 await expect(page.locator('#mode-badge')).toHaveText('ANALYSIS',{timeout:90000});
 const result=await (await page.request.get('/api/jobs/'+job.id)).json();expect(result.status).toBe('completed');
 expect(result.flux_monitors[0].flux.some(v=>Math.abs(v)>0)).toBe(true);
 if(process.env.TORCHFDTD_TEST_CUDA)expect(result.summary.gpu).toContain('5880');
 expect(errors).toEqual([]);
});
