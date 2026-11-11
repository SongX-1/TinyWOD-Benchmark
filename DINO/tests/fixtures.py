import json
from pathlib import Path
from PIL import Image,ImageDraw


def make_dataset(root):
    root=Path(root)
    (root/'annotations').mkdir(parents=True,exist_ok=True)
    categories=[{'id':0,'name':'ignored'}]+[{'id':i+1,'name':n} for i,n in enumerate(
        ['target','target','target','target','target'])]
    for split,count in [('train',3),('val',2),('test',1)]:
        images=[];annotations=[]
        for index in range(1,count+1):
            width,height=(160,160) if index%2 else (160,128)
            image=Image.new('RGB',(width,height),(30+index*10,100,150))
            draw=ImageDraw.Draw(image)
            name=f'样本_{index}.png'
            folder=root/'images'/split;folder.mkdir(parents=True,exist_ok=True)
            images.append({'id':index,'file_name':name,'width':width,'height':height})
            if index!=2:
                for c in range(1,6):
                    x,y=8+c*20,20+c*12
                    w,h=(10,12) if c<4 else (34,34)
                    draw.rectangle([x,y,x+w,y+h],fill=(230,20+c*30,40))
                    annotations.append({'id':len(annotations)+1,'image_id':index,'category_id':c,
                                        'bbox':[x,y,w,h],'area':w*h,'iscrowd':0})
            image.save(folder/name)
        if split!='test':
            (root/'annotations'/f'instances_{split}.json').write_text(json.dumps(
                {'info':{},'images':images,'annotations':annotations,'categories':categories},ensure_ascii=False),encoding='utf-8')
    return root
