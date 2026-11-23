import subprocess, sys, json
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
W,H,FPS,TOTAL=1080,1920,30,30
B="/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"; R="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
F=lambda p,s: ImageFont.truetype(p,s)
meta=json.load(open("img/meta.json"))
# start,end,image,title,subtitle,accent,zoom_dir,pan(x frac)
S=[(0,2.6,"kod","Denizli'de\nboş vaktin\nmi var?","",(255,205,60),1,0.5),
   (2.6,5,"grup5","DENİZLİ\nGENÇLİK\nMERKEZİ","Gençlik ve Spor Bakanlığı",(255,255,255),-1,0.45),
   (5,8,"resim3","Resim &\nSanat","Fırçanı al, gel!",(255,120,160),1,0.6),
   (8,11,"muzik","Müzik","Sahne seni bekliyor",(120,220,255),-1,0.55),
   (11,14,"robot","Robotik &\nTeknoloji","Geleceği sen kodla",(130,255,170),1,0.5),
   (14,17,"tiyatro4","Tiyatro","Rolünü kap!",(255,170,80),-1,0.45),
   (17,20,"satranc","Akıl & Zekâ\nOyunları","Hamle sırası sende",(255,230,120),1,0.5),
   (20,23,"masatenisi2","Spor","Enerjini at!",(120,200,255),-1,0.5),
   (23,26,"grup5","Yeni\nArkadaşlar","Birlikte daha güzel",(255,205,60),1,0.5),
   (26,30,"resim2","","",(255,205,60),1,0.5)]
def cover(name,scale=1.18):
    im=Image.open(f"img/{name}.jpg").convert("RGB"); tw,th=int(W*scale),int(H*scale)
    r=max(tw/im.width,th/im.height); im=im.resize((int(im.width*r)+1,int(im.height*r)+1),Image.LANCZOS)
    return im
imgs={s[2]:cover(s[2]) for s in S}
grad=Image.new("RGBA",(W,H)); gd=ImageDraw.Draw(grad)
for y in range(H):
    a=int(max(0,(y-H*0.35)/(H*0.65))**1.4*215)+ (int((1-y/(H*0.18))*120) if y<H*0.18 else 0)
    gd.line((0,y,W,y),fill=(0,0,0,a))
ease=lambda t: 1-(1-max(0,min(1,t)))**3
def credit(name):
    m=meta[name]; a=m["artist"]
    a=a if len(a)<40 else a[:38]+"…"
    return f"Foto: {a} / {m['license']} – Wikimedia Commons"
p=subprocess.Popen(["ffmpeg","-y","-loglevel","error","-f","rawvideo","-pix_fmt","rgb24","-s",f"{W}x{H}","-r",str(FPS),"-i","-",
  "-f","lavfi","-i","anullsrc=r=44100:cl=stereo","-shortest","-c:v","libx264","-preset","medium","-crf","19",
  "-pix_fmt","yuv420p","-c:a","aac","-movflags","+faststart",sys.argv[1]],stdin=subprocess.PIPE)
cf=F(R,22); foot=F(B,34)
for i in range(TOTAL*FPS):
    t=i/FPS
    k=next(j for j,s in enumerate(S) if s[0]<=t<s[1] or j==len(S)-1)
    st,en,name,title,sub,acc,zd,px=S[k]; lt=t-st; dur=en-st; u=lt/dur
    src=imgs[name]; z=1.0+0.15*(u if zd>0 else 1-u)
    cw,ch=W*1.18/z,H*1.18/z
    cx=(src.width-cw)*(px+(0.08*(u-0.5)*zd)); cx=max(0,min(src.width-cw,cx)); cy=(src.height-ch)/2
    fr=src.crop((int(cx),int(cy),int(cx+cw),int(cy+ch))).resize((W,H),Image.BILINEAR)
    last=(k==len(S)-1)
    if last or k==1:
        fr=fr.filter(ImageFilter.GaussianBlur(18)); fr=ImageEnhance.Brightness(fr).enhance(0.45)
    if k==0: fr=ImageEnhance.Brightness(fr).enhance(0.6)
    fr=ImageEnhance.Color(fr).enhance(1.15).convert("RGBA")
    fr=Image.alpha_composite(fr,grad)
    L=Image.new("RGBA",(W,H),(0,0,0,0)); d=ImageDraw.Draw(L)
    if title:
        lines=title.split("\n"); sz=170 if k!=1 else 160
        while max(d.textlength(l,font=F(B,sz)) for l in lines)>W-140: sz-=4
        tf=F(B,sz); lh=int(sz*1.08)
        y0=(H//2-len(lines)*lh//2-80) if k in (0,1) else H-560-len(lines)*lh
        for n,l in enumerate(lines):
            la=ease((lt-0.05-n*0.1)/0.35); x=70+int(-120*(1-la)) if k not in (0,1) else (W-d.textlength(l,font=tf))/2
            yy=y0+n*lh+(int(50*(1-la)) if k in (0,1) else 0)
            d.text((x+5,yy+6),l,font=tf,fill=(0,0,0,int(140*la)))
            d.text((x,yy),l,font=tf,fill=acc+(int(255*la),))
        yb=y0+len(lines)*lh+25
        if sub:
            sa=ease((lt-0.35)/0.35); sf=F(B,50); w=d.textlength(sub,font=sf)
            x=70 if k not in (0,1) else (W-w)/2-35
            d.rounded_rectangle((x,yb,x+w+70,yb+92),46,fill=acc+(int(235*sa),))
            d.text((x+35,yb+16),sub,font=sf,fill=(20,20,40,int(255*sa)))
    if last:
        info=[("Seni de\nbekliyoruz!",None)]
        tf=F(B,150); lines=["Seni de","bekliyoruz!"]; sz=150
        while max(d.textlength(l,font=F(B,sz)) for l in lines)>W-140: sz-=4
        tf=F(B,sz)
        for n,l in enumerate(lines):
            la=ease((lt-n*0.12)/0.4); w=d.textlength(l,font=tf)
            d.text(((W-w)/2,520+n*int(sz*1.1)+int(50*(1-la))),l,font=tf,fill=(255,205,60,int(255*la)))
        rows=[("İstiklal Cd. No:164","Pamukkale / Denizli"),("0258 241 07 46",""),("@gsb_denizligm","")]
        yy=960
        for n,(a,b) in enumerate(rows):
            la=ease((lt-0.6-n*0.25)/0.4); f=F(B,58) if n==2 else F(R,54)
            w=d.textlength(a,font=f)
            if n==2:
                d.rounded_rectangle(((W-w)/2-50,yy-10,(W+w)/2+50,yy+90),50,fill=(255,205,60,int(255*la)))
                d.text(((W-w)/2,yy+8),a,font=f,fill=(15,25,70,int(255*la)))
            else:
                d.text(((W-w)/2,yy),a,font=f,fill=(255,255,255,int(255*la)))
                if b: w2=d.textlength(b,font=F(R,44)); d.text(((W-w2)/2,yy+70),b,font=F(R,44),fill=(230,230,230,int(255*la))); yy+=60
            yy+=120
    # progress bar, footer, credit
    d.rounded_rectangle((60,80,W-60,90),5,fill=(255,255,255,70))
    d.rounded_rectangle((60,80,60+int((W-120)*t/TOTAL),90),5,fill=(255,255,255,230))
    ft="GSB DENİZLİ GENÇLİK MERKEZİ"; w=d.textlength(ft,font=foot); d.text(((W-w)/2,H-250),ft,font=foot,fill=(255,255,255,200))
    c=credit(name); w=d.textlength(c,font=cf); d.text(((W-w)/2,H-190),c,font=cf,fill=(255,255,255,150))
    out=Image.alpha_composite(fr,L).convert("RGB")
    # white flash at cuts
    if k>0 and lt<0.12: out=Image.blend(out,Image.new("RGB",(W,H),"white"),0.55*(1-lt/0.12))
    p.stdin.write(out.tobytes())
    if i==int(6.5*FPS): out.save(sys.argv[2],quality=92)
p.stdin.close(); p.wait()
