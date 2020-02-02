Shader "Segmentation"
{
	Properties
	{
		_MainTex("Albedo (RGB)", 2D) = "white" {}
		_SegColor("Segmentation Color", Color) = (0.0235,0.8984,0.8984,1)
	}
	SubShader
	{
		Tags { "RenderType" = "Opaque" }
		Pass
	{
	Cull Off
	CGPROGRAM
	#pragma vertex vert
	#pragma fragment frag
	#include "UnityCG.cginc"
	struct v2f {
		float4 pos : SV_POSITION;
		float4 worldPos : NORMAL;
	};

	fixed4 _SegColor;
	v2f vert(float4 vertex : POSITION, float3 normal : NORMAL)
	{
		v2f o;
		o.pos = UnityObjectToClipPos(vertex);
		o.worldPos = mul(unity_ObjectToWorld, vertex);
		return o;
	}

	fixed4 frag(v2f i) : SV_Target
	{
		return fixed4(_SegColor.rgb, 1.0);
	}
	ENDCG
}
	}
}
