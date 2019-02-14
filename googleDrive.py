from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os
from apiclient import errors

class GoogleDriveInterface:
    def __init__(self):
        gauth = GoogleAuth()
        # Try to load saved client credentials
        gauth.LoadCredentialsFile("mycreds.txt")
        if gauth.credentials is None:
            # Authenticate if they're not there
            gauth.LocalWebserverAuth()
        elif gauth.access_token_expired:
            # Refresh them if expired
            gauth.Refresh()
        else:
            # Initialize the saved creds
            gauth.Authorize()
        # Save the current credentials to a file
        gauth.SaveCredentialsFile("mycreds.txt")
        self.drive = GoogleDrive(gauth)
        self.creds = gauth

    def uploadFile(self, path, folder_id="'1xOe202CuYu5VpPsRitoT0jFulHDSUnPh'"):
        file_name = os.path.basename(path)
        print('Uploading ', file_name)
        with open(path,"r") as file:
            children = self.drive.ListFile({'q': folder_id + " in parents and trashed=false"}).GetList()
            for child in children:
                if (child['originalFilename'] == file_name):
                    child.SetContentString(file.read())
                    child.Upload()
                    print('Updated ', file_name)
                    return
            file_drive = self.drive.CreateFile({'title': file_name, "parents":  [{"id": folder_id[1:-1]}]  })  
            file_drive.SetContentString(file.read())
            file_drive.Upload()
            print('Uploaded ', file_name)
    def downloadFile(self, path, folder_id="'1xOe202CuYu5VpPsRitoT0jFulHDSUnPh'"):
        file_name = os.path.basename(path)
        print('Requesting ', file_name)
        children = self.drive.ListFile({'q': folder_id + " in parents and trashed=false"}).GetList()
        for child in children:
            if (child['originalFilename'] == file_name):
                file = self.drive.CreateFile({'id': child['id']})
                file.GetContentFile(path)
                print('Downloaded ', file_name)
                return True
        print('Not available to download ', file_name)
        return False