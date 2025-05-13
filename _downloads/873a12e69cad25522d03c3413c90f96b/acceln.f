c**********************************************************************
c
c     calculate acceleration on moving body
c
c     copyright daresbury laboratory
c     author w.smith may 2005
c
c**********************************************************************
      implicit real*8(a-h,o-z)

c     skip header records in XY file

      read(*,*)
      read(*,*)
      read(*,*)
      read(*,*)
      write(*,'(a)')'# Nanoswitch Project'
      write(*,'(a)')'# Nanotube Acceleration'
      write(*,'(a)')'# Time (ps)'
      write(*,'(a)')'# Acc (A/ps^2)'

c     read data from XY file

      read(*,*)x0,y0
      read(*,*)x1,y1
      dt2=(x1-x0)**2
      do while(.true.)

        read(*,*,end=100)x2,y2
        acc=(y2-2.d0*y1+y0)/dt2
        write(*,'(1p,2e14.6)')x1,acc
        y0=y1
        y1=y2
        x1=x2

      enddo
 100  continue
      end
