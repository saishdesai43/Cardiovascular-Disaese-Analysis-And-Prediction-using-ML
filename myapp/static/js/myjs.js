function check()
{
    
    var u=document.getElementById("uname").value;
    var p=document.getElementById("pwd").value;
    alert("Hi "+u+" "+p +" check ");
    if((u=="")&&(p==""))
    {
        alert("Username or password cant be blank");
    }
}