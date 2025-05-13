!**********************************************************************
!  history file from DL_POLY_4 to xyz 
!
! © Daresbury Laboratory
! author: Alin M Elena <alin-marin.elena@stfc.ac.uk>
!
!**********************************************************************
program comhis
  use iso_fortran_env 
  implicit none
  character(len=100) :: line, title,dummy,arg
  character(len=6) :: el
  integer :: stat
  integer :: i,j,timestep,nt1,id1,id2,id3
  integer :: itrj,icon,natms,nframes,nrec
  real(kind=8) ::  x, y, z
  real(kind=8) :: vx,vy,vz
  real(kind=8) :: fx,fy,fz
  real(kind=8) :: cx,cy,cz
  real(kind=8) :: dt,t

  if (command_argument_count() /= 1 ) then 
    write(*,*) "Incorrect usage!!!"
    call  get_command_argument(0, arg)
    write(*,*) "correct usage: ", trim(arg), " <noOfAtoms>"
    stop
  endif
  call  get_command_argument(1, arg)
  read(arg,*) nt1

  open(101,file="HISTORY",action="read")
  open(102,file="COM.XY",action="write",status="unknown")
  write(102,*)"# Nanoswitch model"
  write(102,*)"# Centre of mass motion"
  write(102,*)"# "
  write(102,*)'# Time [ps] | Position [Å]'
  read(101,'(a100)') line
  title = line
  read(101,'(a100)') line
  read(line,*)itrj,icon,natms,nframes,nrec
  write(*,*)"frames: ",nFrames," atoms ", natms, " skipping first ", nt1, " atoms"
  do i=1,nframes
    read(101,'(a100)') line
    read(line,*)dummy,timestep,id1,id2,id3,dt,t
    read(101,'(a100)') line
    read(101,'(a100)') line
    read(101,'(a100)') line
    cx=0.0_8
    do j=1,natms
      read(101,'(a100)') line
      read(101,'(a100)') line
      read(line,*) x,y,z 
      if (j>nt1) then 
        cz=cz+z
      endif
      if (itrj == 1) then
        read(101,'(a100)') line
        read(line,*) vx,vy,vz 
      endif
      if (itrj == 2) then
        read(101,'(a100)') line
        read(line,*) vx,vy,vz 
        read(101,'(a100)') line
        read(line,*) fx,fy,fz 
      endif
    enddo
    cz=cz/(natms-nt1)
    write(102,*)t,cz
  enddo
  close(101) 
  close(102) 
end program comhis
