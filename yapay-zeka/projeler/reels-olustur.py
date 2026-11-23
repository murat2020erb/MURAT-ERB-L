import subprocess, math, sys
from PIL import Image, ImageDraw, ImageFont, ImageFilter
W,H,FPS=1080,1920,30
B="/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"; R="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
F=lambda p,s: ImageFont.truetype(p,s)
# (start,end,title,subtitle,color_top,color_bottom,accent)
S=[(0,2.5,"Denizli'de\nboş vaktin\nmi var?","",(20,24,60),(60,20,90),(255,200,60)),
   (2.5,5,"DENİZLİ\nGENÇLİK\nMERKEZİ","Gençlik ve Spor Bakanlığı",(10,40,100),(0,120,200),(255,255,255)),
   (5,9,"Kurslar\n&\nAtölyeler","Yeni bir şey öğren",(90,20,110),(220,60,120),(255,220,90)),
   (9,13,"Tiyatro\nŞiir\nMüzik","Sahne senin",(150,40,30),(240,120,40),(255,240,200)),
   (13,17,"Yeni\narkadaşlar","Aynı ilgi alanı, yeni dostluklar",(0,90,80),(30,170,130),(230,255,240)),
   (17,21,"Bilgi\nyarışmaları","Bil, yarış, kazan",(30,30,90),(90,70,200),(140,220,255)),
   (21,25,"Gençlerin\nbuluşma\nnoktası","Denizli'nin kalbinde",(120,20,60),(200,40,80),(255,210,220)),
   (25,30,"Seni de\nbekliyoruz!","",(10,40,100),(0,130,210),(255,200,60))]
def grad(c1,c2):
    g=Image.new("RGB",(1,H))
    for y in range(H):
        t=y/(H-1); g.putpixel((0,y),tuple(int(c1[i]+(c2[i]-c1[i])*t) for i in range(3)))
    g=g.resize((W,H))
    d=ImageDraw.Draw(g,"RGBA")
    d.ellipse((-300,-200,700,800),fill=(255,255,255,18)); d.ellipse((500,1300,1500,2300),fill=(255,255,255,14))
    return g
bgs=[grad(s[4],s[5]) for s in S]
ease=lambda t: 1-(1-max(0,min(1,t)))**3
TOTAL=30
p=subprocess.Popen(["ffmpeg","-y","-f","rawvideo","-pix_fmt","rgb24","-s",f"{W}x{H}","-r",str(FPS),"-i","-",
  "-f","lavfi","-i","anullsrc=r=44100:cl=stereo","-shortest",
  "-c:v","libx264","-preset","medium","-crf","20","-pix_fmt","yuv420p","-c:a","aac","-movflags","+faststart",sys.argv[1]],stdin=subprocess.PIPE)
tf=F(B,150); sf=F(R,52); small=F(B,44); foot=F(R,40)
for i in range(TOTAL*FPS):
    t=i/FPS
    k=next(j for j,s in enumerate(S) if s[0]<=t<s[1] or j==len(S)-1)
    st,en,title,sub,_,_,acc=S[k]; lt=t-st
    bg=bgs[k]; z=1+0.04*(lt/(en-st))
    img=bg.resize((int(W*z),int(H*z))).crop((int((W*z-W)/2),int((H*z-H)/2),int((W*z-W)/2)+W,int((H*z-H)/2)+H))
    layer=Image.new("RGBA",(W,H),(0,0,0,0)); d=ImageDraw.Draw(layer)
    a=ease(lt/0.45); out=1-ease((lt-(en-st-0.25))/0.25) if k<len(S)-1 else 1
    alpha=int(255*a*out); dy=int(80*(1-a))
    lines=title.split("\n"); sz=150
    while max(d.textlength(l,font=F(B,sz)) for l in lines)>W-160: sz-=4
    tf=F(B,sz); lh=int(sz*1.13); y0=H//2-len(lines)*lh//2-120+dy
    for n,l in enumerate(lines):
        la=ease((lt-n*0.12)/0.4); w=d.textlength(l,font=tf)
        d.text(((W-w)/2+4,y0+n*lh+int(60*(1-la))+6),l,font=tf,fill=(0,0,0,int(90*la*out)))
        d.text(((W-w)/2,y0+n*lh+int(60*(1-la))),l,font=tf,fill=acc+(int(255*la*out),))
    yb=y0+len(lines)*lh+40
    bw=int(260*ease((lt-0.3)/0.4)); d.rounded_rectangle(((W-bw)/2,yb,(W+bw)/2,yb+12),6,fill=acc+(alpha,))
    if sub:
        w=d.textlength(sub,font=sf); d.text(((W-w)/2,yb+50),sub,font=sf,fill=(255,255,255,alpha))
    if k==len(S)-1:
        info=["İstiklal Cd. No:164, Pamukkale","0258 241 07 46","@gsb_denizligm"]
        for n,l in enumerate(info):
            la=ease((lt-0.6-n*0.25)/0.4); f=small if n==2 else sf
            w=d.textlength(l,font=f); yy=yb+70+n*95+int(40*(1-la))
            if n==2:
                d.rounded_rectangle(((W-w)/2-40,yy-18,(W+w)/2+40,yy+66),40,fill=(255,200,60,int(255*la)))
                d.text(((W-w)/2,yy),l,font=f,fill=(20,30,80,int(255*la)))
            else: d.text(((W-w)/2,yy),l,font=f,fill=(255,255,255,int(255*la)))
    # top progress bar + footer
    d.rounded_rectangle((60,90,W-60,100),5,fill=(255,255,255,60))
    d.rounded_rectangle((60,90,60+int((W-120)*t/TOTAL),100),5,fill=(255,255,255,220))
    ft="GSB Denizli Gençlik Merkezi"; w=d.textlength(ft,font=foot); d.text(((W-w)/2,H-160),ft,font=foot,fill=(255,255,255,170))
    frame=Image.alpha_composite(img.convert("RGBA"),layer).convert("RGB")
    p.stdin.write(frame.tobytes())
    if i==int(3.5*FPS): frame.save(sys.argv[2])
p.stdin.close(); p.wait()
