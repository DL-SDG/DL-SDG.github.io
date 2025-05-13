c**********************************************************************
c
c     fourier tranform program
c
c     copyright daresbury laboratory 1998
c     author w.smith july 1998
c
c**********************************************************************

      implicit real*8 (a-h,o-z)

      parameter (ndiv=4096)

      complex*16 aaa(ndiv),wfft(ndiv)
      dimension key(ndiv),bbb(2,ndiv)

      data tpi/6.2831853072d0/

c     set up fft array

      np2=ndiv/2
      tpn=tpi/dble(ndiv)
      do i=1,ndiv
        aaa(i)=(0.d0,0.d0)
      enddo

c     skip header records in XY file

      read(*,*)
      read(*,*)
      read(*,*)
      read(*,*)

c     read data from XY file

      nnn=0

      sum=0.d0
      do while(nnn.lt.ndiv)

        read(*,*,end=100)x,y
        if(nnn.eq.0)x0=x
        nnn=nnn+1
        aaa(nnn)=cmplx(y,0.d0)
        sum=sum+y
        xn=x

      enddo

 100  continue

c     subtract average value

      ave=sum/dble(nnn)

      do i=1,nnn
        aaa(i)=aaa(i)-ave
      enddo

      if(nnn.lt.ndiv)then
        aaa(1)=0.5d0*aaa(1)
        aaa(nnn)=0.5d0*aaa(nnn)
      endif

c     define frequency interval

      df=dble(nnn-1)/((xn-x0)*dble(ndiv))

c     apply gaussian window function

      arg=14.d0/dble(np2)**2
      do i=1,ndiv
         aaa(i)=aaa(i)*exp(-arg*dble(i-np2)**2)
      enddo

c     initialise the fft routine

      call fft(1,1,ndiv,key,aaa,wfft,bbb)

c     perform FT of function

      call fft(0,1,ndiv,key,aaa,wfft,bbb)

c     print result

      write(*,'(a)')'# Nanoswitch dynamics project'
      write(*,'(a)')'# Fourier transform of COM motion'
      write(*,'(a)')'# Frequency (1/ps)'
      write(*,'(a)')'# FT of position'
      do i=1,100
        write(*,'(1p,2e12.4)')df*dble(i-1),
     x    sqrt(bbb(1,i)**2+bbb(2,i)**2)
      enddo

      end
      subroutine fft(ind,isw,ndiv,key,aaa,wfft,bbb)
c***********************************************************************
c     
c     fast fourier transform routine
c     
c     copyright daresbury laboratory 1994
c
c     author w smith
c
c***********************************************************************
      
      implicit real*8(a-h,o-z)
      
      logical check
      complex*16 aaa(ndiv),bbb(ndiv),wfft(ndiv),ttt
      dimension key(ndiv)
      data tpi/6.2831853072d0/
   10 format(1h0,'error - number of points not a power of two')
      
c     
c     check that array is of suitable length
      nt=1
      check=.true.
      do i=1,20
         nt=2*nt
         if(nt.eq.ndiv)then
            check=.false.
            nu=i
         endif
      enddo
      if(check)then
         write(*,10)
         stop
      endif
      
      if(ind.gt.0)then
c     
c     set reverse bit address array
         
         do kkk=1,ndiv
            iii=0
            jjj=kkk-1
            do j=1,nu
               jj2=jjj/2
               iii=2*(iii-jj2)+jjj
               jjj=jj2
            enddo
            key(kkk)=iii+1
         enddo
c     
c     initialise complex exponential factors
         
         np1=ndiv+1
         np2=ndiv/2
         tpn=tpi/dble(ndiv)
         wfft(1)=(1.d0,0.d0)
         do i=1,np2
            arg=tpn*dble(i)
            wfft(i+1)=cmplx(cos(arg),sin(arg))
            wfft(np1-i)=conjg(wfft(i+1))
         enddo
         
         return
      endif
      
c     
c     take conjugate of exponentials if required
      
      if(isw.lt.0)then
         
         do i=1,ndiv
            wfft(i)=conjg(wfft(i))
         enddo
         
      endif
      
c     
c     take copy input array
      
      do i=1,ndiv
         bbb(i)=aaa(i)
      enddo
c     
c     perform fourier transform
      
      kkk=0
      np2=ndiv/2
      do l=1,nu

  100    do i=1,np2
            iii=key(kkk/np2+1)
            kk1=kkk+1
            k12=kk1+np2
            ttt=bbb(k12)*wfft(iii)
            bbb(k12)=bbb(kk1)-ttt
            bbb(kk1)=bbb(kk1)+ttt
            kkk=kkk+1
         enddo
         kkk=kkk+np2
         if(kkk.lt.ndiv)go to 100
         kkk=0
         np2=np2/2

      enddo
c     
c     unscramble the fft using bit address array
      
      do kkk=1,ndiv
         iii=key(kkk)
         if(iii.gt.kkk)then
            ttt=bbb(kkk)
            bbb(kkk)=bbb(iii)
            bbb(iii)=ttt
         endif
      enddo
c     
c     restore exponentials to unconjugated values if necessary
      
      if(isw.lt.0)then
         
         do i=1,ndiv
            wfft(i)=conjg(wfft(i))
         enddo
         
      endif
      
      return
      end
