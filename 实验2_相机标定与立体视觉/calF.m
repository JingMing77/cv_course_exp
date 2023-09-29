function F=calF(R,T,Kl,Kr)
Tx = zeros(3);
Tx(3,2)=T(1);Tx(2,3)=-T(1);
Tx(1,3)=T(2);Tx(3,1)=-T(2);
Tx(2,1)=T(3);Tx(1,2)=-T(3);
E = Tx*R;
F = (Kl.')\E/Kr;
end